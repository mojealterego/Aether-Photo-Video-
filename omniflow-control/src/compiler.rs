#![forbid(unsafe_code)]
use crate::ir::*;
use petgraph::algo::toposort;
use petgraph::graph::{DiGraph, NodeIndex};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum CompileError {
    #[error("unsupported or empty OmniGraph schema")]
    Schema,
    #[error("graph exceeds configured node budget")]
    NodeBudget,
    #[error("unknown operator, node or model: {0}")]
    Unresolved(String),
    #[error("duplicate input binding: {0}")]
    DuplicateInput(String),
    #[error("type mismatch: {0}")]
    SemanticTypeMismatch(String),
    #[error("illegal cycle in graph")]
    Cycle,
    #[error("untrusted worker lacks sandbox: {0}")]
    Isolation(String),
    #[error("missing explicit side-effect authorisation: {0}")]
    SideEffect(String),
    #[error("license or model policy denied: {0}")]
    ModelDenied(String),
    #[error("documentary identity provenance gate rejected")]
    IdentityGate,
    #[error("VRAM or resource budget exceeded")]
    ResourceBudget,
    #[error("numeric resource overflow")]
    ArithmeticOverflow,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPlan {
    pub graph_digest: String,
    pub topological_order: Vec<String>,
    pub predicted_peak_bytes: u64,
    pub estimated_gpu_seconds: u64,
    pub node_count: usize,
}

#[derive(Default)]
pub struct OmniGraphCompiler {
    pub node_registry: BTreeMap<(String, String), NodeDefinition>,
    pub model_registry: BTreeMap<String, ModelManifest>,
}
impl OmniGraphCompiler {
    pub fn new() -> Self { Self::default() }
    pub fn register_node(&mut self, def: NodeDefinition) {
        self.node_registry.insert((def.op_id.clone(), def.op_version.clone()), def);
    }
    pub fn register_model(&mut self, model: ModelManifest) {
        self.model_registry.insert(model.model_id.clone(), model);
    }

    pub fn compile(&self, graph: &GraphSpec, policy: &JobPolicy) -> Result<ExecutionPlan, CompileError> {
        if graph.schema_version != "omnigraph.v1.1"
            || graph.graph_id.is_empty() || graph.output_targets.is_empty()
            || policy.tenant_id.is_empty() || policy.principal_id.is_empty() {
            return Err(CompileError::Schema);
        }
        if graph.nodes.is_empty() || graph.nodes.len() > policy.max_nodes {
            return Err(CompileError::NodeBudget);
        }
        if graph.output_targets.iter().any(|id| !graph.nodes.contains_key(id)) {
            return Err(CompileError::Unresolved("output target".into()));
        }

        let mut nodes = DiGraph::<String, ()>::new();
        let mut indices: BTreeMap<String, NodeIndex> = BTreeMap::new();
        let mut model_checksums = BTreeMap::new();
        let mut total_peak = 256u64 * 1024 * 1024; // baseline runtime reservation
        let mut gpu_millis = 0u64;

        for (id, node) in &graph.nodes {
            let definition = self.node_registry
                .get(&(node.op_id.clone(), node.op_version.clone()))
                .ok_or_else(|| CompileError::Unresolved(node.op_id.clone()))?;
            if definition.trust_level != TrustLevel::T0CoreTrusted {
                return Err(CompileError::Isolation(id.clone()));
            }
            if matches!(definition.mode, NodeMode::ExternalSideEffect | NodeMode::HumanApproval)
                && (node.approved_actor.as_deref().unwrap_or("").is_empty()
                    || node.reconciler_token.as_deref().unwrap_or("").is_empty()) {
                return Err(CompileError::SideEffect(id.clone()));
            }
            if !policy.allow_biometric_egress
                && definition.mode == NodeMode::ExternalSideEffect {
                return Err(CompileError::SideEffect(id.clone()));
            }
            if policy.fidelity_mode == FidelityMode::Doc
                && node.config_params.get("quantization_bits").and_then(|v| v.as_u64()) == Some(4) {
                return Err(CompileError::SemanticTypeMismatch("DOC disallows unvalidated INT4".into()));
            }
            if let Some(model_id) = &definition.required_model_id {
                if !policy.allowed_models.contains(model_id) {
                    return Err(CompileError::ModelDenied(model_id.clone()));
                }
                let model = self.model_registry.get(model_id)
                    .ok_or_else(|| CompileError::ModelDenied(model_id.clone()))?;
                if (policy.commercial_use && !model.commercial_use_allowed)
                    || model.sha256.len() != 71 || !model.sha256.starts_with("sha256:") {
                    return Err(CompileError::ModelDenied(model_id.clone()));
                }
                model_checksums.insert(model_id, &model.sha256);
            }
            let slot = nodes.add_node(id.clone());
            indices.insert(id.clone(), slot);
            if definition.resources.requires_gpu {
                gpu_millis = gpu_millis.checked_add(definition.resources.expected_gpu_millis)
                    .ok_or(CompileError::ArithmeticOverflow)?;
                let mut step = definition.resources.resident_weights_bytes
                    .checked_add(definition.resources.workspace_bytes)
                    .ok_or(CompileError::ArithmeticOverflow)?;
                for port in definition.output_ports.values() {
                    step = step.checked_add(port.estimated_bytes().ok_or(CompileError::ArithmeticOverflow)?)
                        .ok_or(CompileError::ArithmeticOverflow)?;
                }
                total_peak = total_peak.checked_add(step).ok_or(CompileError::ArithmeticOverflow)?;
            }
        }

        let mut bound = BTreeSet::new();
        let mut outgoing: HashMap<String, Vec<String>> = HashMap::new();
        for edge in &graph.edges {
            let src = indices.get(&edge.source_node)
                .ok_or_else(|| CompileError::Unresolved(edge.source_node.clone()))?;
            let dst = indices.get(&edge.target_node)
                .ok_or_else(|| CompileError::Unresolved(edge.target_node.clone()))?;
            if !bound.insert((edge.target_node.clone(), edge.target_port.clone())) {
                return Err(CompileError::DuplicateInput(edge.target_node.clone()));
            }
            let from = &graph.nodes[&edge.source_node];
            let to = &graph.nodes[&edge.target_node];
            let source = &self.node_registry[&(from.op_id.clone(), from.op_version.clone())];
            let target = &self.node_registry[&(to.op_id.clone(), to.op_version.clone())];
            let source_type = source.output_ports.get(&edge.source_port)
                .ok_or_else(|| CompileError::Unresolved(edge.source_port.clone()))?;
            let target_type = target.input_ports.get(&edge.target_port)
                .ok_or_else(|| CompileError::Unresolved(edge.target_port.clone()))?;
            if source_type != target_type {
                return Err(CompileError::SemanticTypeMismatch(format!(
                    "{}.{} -> {}.{}",
                    edge.source_node, edge.source_port, edge.target_node, edge.target_port
                )));
            }
            nodes.add_edge(*src, *dst, ());
            outgoing.entry(edge.source_node.clone()).or_default().push(edge.target_node.clone());
        }
        let order = toposort(&nodes, None).map_err(|_| CompileError::Cycle)?;
        let ordered: Vec<String> = order.iter().map(|id| nodes[*id].clone()).collect();

        if policy.fidelity_mode == FidelityMode::Doc && policy.require_identity_guard {
            let guards: Vec<String> = graph.nodes.iter()
                .filter(|(_, node)| node.op_id == "F007_ArcFaceIdentityGuard")
                .map(|(id, _)| id.clone()).collect();
            let targets: Vec<String> = graph.output_targets.iter()
                .filter(|id| graph.nodes[*id].op_id == "I019_BigTIFFWriter")
                .cloned().collect();
            if guards.is_empty() || targets.is_empty() { return Err(CompileError::IdentityGate) }
            for target in targets {
                let mut reached = false;
                for guard in &guards {
                    let mut seen = BTreeSet::new();
                    let mut queue = VecDeque::from([guard.clone()]);
                    while let Some(id) = queue.pop_front() {
                        if !seen.insert(id.clone()) { continue }
                        if id == target { reached = true; break }
                        if let Some(next) = outgoing.get(&id) {
                            queue.extend(next.iter().cloned());
                        }
                    }
                    if reached { break }
                }
                if !reached { return Err(CompileError::IdentityGate) }
            }
        }

        if total_peak > policy.available_vram_bytes.saturating_mul(9) / 10
            || gpu_millis.div_ceil(1000) > policy.max_gpu_seconds {
            return Err(CompileError::ResourceBudget);
        }

        let canonical = serde_json::to_vec(&(graph, policy, &model_checksums, &ordered))
            .map_err(|_| CompileError::Schema)?;
        let mut hasher = Sha256::new();
        hasher.update(b"OMNIGRAPH_V1.1:canonical:");
        hasher.update(&canonical);
        Ok(ExecutionPlan {
            graph_digest: format!("sha256:{}", hex::encode(hasher.finalize())),
            topological_order: ordered,
            predicted_peak_bytes: total_peak,
            estimated_gpu_seconds: gpu_millis.div_ceil(1000),
            node_count: graph.nodes.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn image(space: ColorSpace, tf: TransferFunction) -> PortType {
        PortType::Image { width: 32, height: 32, channels: 3,
            dtype: NumericDtype::UInt16, color_space: space, transfer: tf }
    }
    fn definition(id: &str, source: bool, kind: PortType) -> NodeDefinition {
        NodeDefinition { op_id: id.into(), op_version: "1".into(),
            mode: NodeMode::PureDeterministic, trust_level: TrustLevel::T0CoreTrusted,
            input_ports: if source { BTreeMap::new() } else { BTreeMap::from([("in".into(), kind.clone())]) },
            output_ports: BTreeMap::from([("out".into(), kind)]),
            resources: ResourceRequirement { requires_gpu: false, resident_weights_bytes: 0,
                workspace_bytes: 0, expected_gpu_millis: 0, supports_tiling: false },
            required_model_id: None }
    }
    fn fixture() -> (OmniGraphCompiler, GraphSpec, JobPolicy) {
        let mut c = OmniGraphCompiler::new();
        let ty = image(ColorSpace::SceneLinearSrgb, TransferFunction::Linear);
        c.register_node(definition("SOURCE", true, ty.clone()));
        c.register_node(definition("SINK", false, ty));
        let graph = GraphSpec { schema_version:"omnigraph.v1.1".into(), graph_id: "g".into(),
            nodes: BTreeMap::from([
                ("a".into(), NodeInstance { op_id:"SOURCE".into(), op_version:"1".into(),
                    config_params:BTreeMap::new(), approved_actor:None, reconciler_token:None }),
                ("b".into(), NodeInstance { op_id:"SINK".into(), op_version:"1".into(),
                    config_params:BTreeMap::new(), approved_actor:None, reconciler_token:None }),
            ]),
            edges: vec![GraphEdge { source_node:"a".into(), source_port:"out".into(),
                target_node:"b".into(), target_port:"in".into() }],
            input_artifact_digests:BTreeMap::new(), output_targets:BTreeSet::from(["b".into()]) };
        let policy = JobPolicy { tenant_id:"tenant1".into(), principal_id:"operator".into(),
            fidelity_mode:FidelityMode::Editorial, service_class:ServiceClass::Editorial,
            allowed_models:BTreeSet::new(), commercial_use:false,
            allow_biometric_egress:false, require_identity_guard:false, max_nodes:10,
            max_gpu_seconds:30, available_vram_bytes:2_000_000_000, seed:1 };
        (c, graph, policy)
    }
    #[test]
    fn accepted_graph_is_stable_across_compilations() {
        let (c,g,p) = fixture();
        let one=c.compile(&g,&p).unwrap();
        assert_eq!(one.topological_order, vec!["a","b"]);
        assert_eq!(one.graph_digest,c.compile(&g,&p).unwrap().graph_digest);
    }
    #[test]
    fn rejects_implicit_color_conversion() {
        let (mut c,g,p)=fixture();
        c.register_node(definition("SINK",false,image(ColorSpace::DisplaySrgb,TransferFunction::SrgbGamma)));
        assert!(matches!(c.compile(&g,&p),Err(CompileError::SemanticTypeMismatch(_))));
    }
    #[test]
    fn rejects_cycle() {
        let (mut c,mut g,p)=fixture();
        let ty=image(ColorSpace::SceneLinearSrgb,TransferFunction::Linear);
        c.register_node(definition("SOURCE",false,ty));
        g.edges.push(GraphEdge{source_node:"b".into(),source_port:"out".into(),
            target_node:"a".into(),target_port:"in".into()});
        assert!(matches!(c.compile(&g,&p),Err(CompileError::Cycle)));
    }
    #[test]
    fn rejects_unauthorized_model() {
        let (mut c,g,p)=fixture();
        c.node_registry.get_mut(&("SINK".into(),"1".into())).unwrap().required_model_id=Some("unknown".into());
        assert!(matches!(c.compile(&g,&p),Err(CompileError::ModelDenied(_))));
    }
    #[test]
    fn doc_requires_identity_lineage_not_only_guard_presence() {
        let (mut c, mut g, mut p)=fixture();
        p.fidelity_mode=FidelityMode::Doc; p.require_identity_guard=true;
        g.nodes.get_mut("b").unwrap().op_id="I019_BigTIFFWriter".into();
        let ty=image(ColorSpace::SceneLinearSrgb,TransferFunction::Linear);
        c.register_node(definition("I019_BigTIFFWriter",false,ty.clone()));
        c.register_node(definition("F007_ArcFaceIdentityGuard",true,ty));
        g.nodes.insert("guard".into(),NodeInstance{op_id:"F007_ArcFaceIdentityGuard".into(),
            op_version:"1".into(),config_params:BTreeMap::new(),approved_actor:None,reconciler_token:None});
        assert_eq!(c.compile(&g,&p).unwrap_err(),CompileError::IdentityGate);
        g.edges.push(GraphEdge{source_node:"guard".into(),source_port:"out".into(),
            target_node:"b".into(),target_port:"in".into()});
        assert!(matches!(c.compile(&g,&p),Err(CompileError::DuplicateInput(_))));
    }
    #[test]
    fn forbids_unvalidated_doc_int4() {
        let (c,mut g,mut p)=fixture();
        p.fidelity_mode=FidelityMode::Doc;
        g.nodes.get_mut("b").unwrap().config_params.insert("quantization_bits".into(),serde_json::json!(4));
        assert!(matches!(c.compile(&g,&p),Err(CompileError::SemanticTypeMismatch(_))));
    }
}
