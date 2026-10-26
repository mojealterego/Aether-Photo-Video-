#![forbid(unsafe_code)]
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FidelityMode { Doc, Editorial, Creative, IdentityReview }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ServiceClass { Interactive, Editorial, Production, BatchResearch }
impl ServiceClass {
    pub fn weight(self) -> usize {
        match self {
            Self::Interactive => 8, Self::Editorial => 4,
            Self::Production => 2, Self::BatchResearch => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum NodeMode {
    PureDeterministic, PureSeeded, ImpureIdempotent,
    ExternalSideEffect, HumanApproval
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum TrustLevel { T0CoreTrusted, T1VerifiedPlugin, T2UntrustedOmeganode }

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ColorSpace { SceneLinearSrgb, DisplaySrgb, IccProfiled, AcesCg }
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum TransferFunction { Linear, SrgbGamma, Rec709, PqSt2084, Hlg }
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum NumericDtype { UInt8, UInt16, Float16, Float32 }
impl NumericDtype {
    pub fn bytes(&self) -> u64 {
        match self { Self::UInt8 => 1, Self::UInt16 | Self::Float16 => 2,
                     Self::Float32 => 4 }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DepthUnits { RelativeNormalized, MetricMeters }

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "semantic_kind")]
pub enum PortType {
    Image {
        width: u32, height: u32, channels: u8,
        dtype: NumericDtype, color_space: ColorSpace,
        transfer: TransferFunction,
    },
    Depth { width: u32, height: u32, units: DepthUnits, model_id: String },
    Latent { model_family: String, vae_compatibility_id: String,
             channels: u16, dtype: NumericDtype },
    Embedding { model_id: String, dimensions: u16 },
    Artifact { mime: String },
    Control,
}
impl PortType {
    pub fn estimated_bytes(&self) -> Option<u64> {
        match self {
            Self::Image { width, height, channels, dtype, .. } =>
                u64::from(*width).checked_mul(u64::from(*height))?
                  .checked_mul(u64::from(*channels))?.checked_mul(dtype.bytes()),
            Self::Depth { width, height, .. } =>
                u64::from(*width).checked_mul(u64::from(*height))?.checked_mul(4),
            Self::Latent { channels, dtype, .. } =>
                128u64.checked_mul(128)?.checked_mul(u64::from(*channels))?.checked_mul(dtype.bytes()),
            Self::Embedding { dimensions, .. } =>
                u64::from(*dimensions).checked_mul(4),
            _ => Some(4096),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirement {
    pub requires_gpu: bool,
    pub resident_weights_bytes: u64,
    pub workspace_bytes: u64,
    pub expected_gpu_millis: u64,
    pub supports_tiling: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeDefinition {
    pub op_id: String,
    pub op_version: String,
    pub mode: NodeMode,
    pub trust_level: TrustLevel,
    pub input_ports: BTreeMap<String, PortType>,
    pub output_ports: BTreeMap<String, PortType>,
    pub resources: ResourceRequirement,
    pub required_model_id: Option<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInstance {
    pub op_id: String,
    pub op_version: String,
    pub config_params: BTreeMap<String, serde_json::Value>,
    pub approved_actor: Option<String>,
    pub reconciler_token: Option<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub source_node: String,
    pub source_port: String,
    pub target_node: String,
    pub target_port: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSpec {
    pub schema_version: String,
    pub graph_id: String,
    pub nodes: BTreeMap<String, NodeInstance>,
    pub edges: Vec<GraphEdge>,
    pub input_artifact_digests: BTreeMap<String, String>,
    pub output_targets: BTreeSet<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    pub model_id: String,
    pub sha256: String,
    pub commercial_use_allowed: bool,
    pub license_id: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobPolicy {
    pub tenant_id: String,
    pub principal_id: String,
    pub fidelity_mode: FidelityMode,
    pub service_class: ServiceClass,
    pub allowed_models: BTreeSet<String>,
    pub commercial_use: bool,
    pub allow_biometric_egress: bool,
    pub require_identity_guard: bool,
    pub max_nodes: usize,
    pub max_gpu_seconds: u64,
    pub available_vram_bytes: u64,
    pub seed: u64,
}
