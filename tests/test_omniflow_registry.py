"""Versioned native engine registry contracts: no plugin imports or execution."""
import json
import unittest
from dataclasses import replace
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "omniflow-worker"))
from omega_sdk.registry import EngineDescriptor, Registry, RegistryError, EngineStatus


def engine(op='I002_LoadNEF', version='1.1.0', *, state=EngineStatus.VERIFIED, deps=()):
    return EngineDescriptor(
        op_id=op, op_version=version, abi_version='omega.node.abi.v1.1',
        provider='AETHERIS', family='SOURCE_AND_COLOR',
        trust_class='T0_CORE_TRUSTED', mode='PURE_DETERMINISTIC',
        state=state, provenance_sha256='sha256:'+'a'*64,
        license_id='AETHERIS-PROPRIETARY', dependencies=deps,
        input_ports={}, output_ports={'rgb': 'IMAGE:SCENE_LINEAR:FLOAT32'},
        capabilities=('camera.nef', 'color.linear'),
        test_evidence=('local:unit:2026-10-26',),
    )


class RegistryTests(unittest.TestCase):
    def test_manifest_json_round_trip_and_stable_digest(self):
        one=engine()
        blob=one.canonical_json()
        two=EngineDescriptor.from_mapping(json.loads(blob))
        self.assertEqual(one, two)
        self.assertEqual(one.digest(), two.digest())
        self.assertEqual(one.digest(), 'sha256:'+__import__('hashlib').sha256(blob.encode()).hexdigest())

    def test_add_multiple_versions_idempotently_and_resolve_exactly(self):
        r=Registry()
        a=engine(); b=engine(version='1.2.0')
        r.register(a); r.register(a); r.register(b)
        self.assertEqual(len(r.list_all()),2)
        self.assertEqual(r.resolve(a.op_id,'1.1.0'),a)
        self.assertEqual(r.resolve(a.op_id,'1.2.0'),b)

    def test_conflicting_same_version_rejected(self):
        r=Registry(); a=engine(); r.register(a)
        with self.assertRaises(RegistryError): r.register(replace(a,license_id='GPL-3.0'))

    def test_paused_engine_never_eligible_for_activation(self):
        r=Registry(); r.register(engine('F010_FaceOmega',state=EngineStatus.ON_HOLD))
        with self.assertRaisesRegex(RegistryError,'not eligible'):
            r.resolve('F010_FaceOmega','1.1.0',require_verified=True)

    def test_experimental_is_not_production_ready(self):
        r=Registry(); r.register(engine(state=EngineStatus.EXPERIMENTAL))
        with self.assertRaises(RegistryError):r.resolve('I002_LoadNEF','1.1.0',require_verified=True)

    def test_missing_dep_is_rejected_in_planning(self):
        r=Registry();r.register(engine('PHOTO',deps=(('UNKNOWN','1.0.0'),)))
        with self.assertRaisesRegex(RegistryError,'missing dependency'):
            r.plan(('PHOTO','1.1.0'))

    def test_topological_dependency_order_with_same_module_versions(self):
        r=Registry();r.register(engine('A'));r.register(engine('B',deps=(('A','1.1.0'),)))
        r.register(engine('C',deps=(('B','1.1.0'),('A','1.1.0'))))
        self.assertEqual([x.op_id for x in r.plan(('C','1.1.0'))],['A','B','C'])

    def test_cycles_fail_closed(self):
        r=Registry();r.register(engine('A',deps=(('B','1.1.0'),)))
        r.register(engine('B',deps=(('A','1.1.0'),)))
        with self.assertRaisesRegex(RegistryError,'cycle'):
            r.plan(('A','1.1.0'))

    def test_reject_unknown_fields_and_invalid_semver(self):
        a=engine()
        for version in ('latest','1.0','v1.1.0','1.0.0+local','1.0.0-rc1'):
            with self.subTest(version=version),self.assertRaises(RegistryError):
                EngineDescriptor.from_mapping({**a.to_dict(),'op_version':version})
        with self.assertRaises(RegistryError):
            EngineDescriptor.from_mapping({**a.to_dict(),'execute':'/bin/bash'})

    def test_reject_bad_digest_and_unsafe_identifiers(self):
        a=engine()
        for op in ('../evil','other/path','bad space',''):
            with self.subTest(op=op),self.assertRaises(RegistryError):
                EngineDescriptor.from_mapping({**a.to_dict(),'op_id':op})
        with self.assertRaises(RegistryError):
            EngineDescriptor.from_mapping({**a.to_dict(),'provenance_sha256':'sha256:fake'})

    def test_production_requires_verification_evidence_and_provenance(self):
        a=engine()
        with self.assertRaisesRegex(RegistryError,'evidence'):
            EngineDescriptor.from_mapping({**a.to_dict(),'test_evidence':[]})

    def test_registry_capability_query_filters_paused(self):
        r=Registry();r.register(engine());r.register(engine('SLE_Engine',state=EngineStatus.ON_HOLD))
        got=r.find_capability('camera.nef',require_verified=True)
        self.assertEqual([x.op_id for x in got],['I002_LoadNEF'])

    def test_export_import_catalog_round_trip_fails_closed(self):
        r=Registry();r.register(engine());r.register(engine('B',state=EngineStatus.PLANNED))
        exported=r.to_json()
        copied=Registry.from_json(exported)
        self.assertEqual(exported,copied.to_json())
        with self.assertRaises(RegistryError):
            Registry.from_json('{"version": "missing", "entries": []}')

    def test_immutable_registry_snapshots_resist_external_mutation(self):
        out={'rgb':'IMAGE:SCENE_LINEAR:FLOAT32'}
        raw=replace(engine(),output_ports=out)
        r=Registry();r.register(raw)
        out['rgb']='IMAGE:DISPLAY_SRGB:UINT8'
        self.assertEqual(r.resolve('I002_LoadNEF','1.1.0').output_ports['rgb'],
                         'IMAGE:SCENE_LINEAR:FLOAT32')
        with self.assertRaises(TypeError):
            raw.output_ports['rgb']='IMAGE:DISPLAY_SRGB:UINT8'

    def test_links_require_strict_semantic_port_match(self):
        r=Registry()
        source=engine()
        sink=replace(engine(op='NEXT'),input_ports={'frame':'IMAGE:SCENE_LINEAR:FLOAT32'})
        r.register(source);r.register(sink)
        r.validate_link(('I002_LoadNEF','1.1.0'),'rgb',('NEXT','1.1.0'),'frame')
        alt=replace(engine(op='WRITER'),input_ports={'frame':'IMAGE:DISPLAY_SRGB:UINT16'})
        r.register(alt)
        with self.assertRaisesRegex(RegistryError,'semantic type mismatch'):
            r.validate_link(('I002_LoadNEF','1.1.0'),'rgb',('WRITER','1.1.0'),'frame')

    def test_missing_ports_are_never_silently_created(self):
        r=Registry();r.register(engine());r.register(replace(engine(op='TARGET'),input_ports={}))
        with self.assertRaisesRegex(RegistryError,'missing port'):
            r.validate_link(('I002_LoadNEF','1.1.0'),'rgb',('TARGET','1.1.0'),'frame')

    def test_deep_chain_planning_does_not_exhaust_python_stack(self):
        r=Registry()
        for i in range(1250):
            r.register(engine(f'ENGINE_{i}', deps=((f'ENGINE_{i-1}','1.1.0'),) if i else ()))
        plan=r.plan(('ENGINE_1249','1.1.0'))
        self.assertEqual(len(plan),1250)
        self.assertEqual(plan[0].op_id,'ENGINE_0')
        self.assertEqual(plan[-1].op_id,'ENGINE_1249')

    def test_only_exact_manifest_abi_is_admissible(self):
        a=engine()
        with self.assertRaises(RegistryError):
            EngineDescriptor.from_mapping({**a.to_dict(),'abi_version':'omega.node.abi.v1'})

if __name__ == '__main__': unittest.main()
