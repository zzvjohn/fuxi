"""Run with python -m fuxi_v1; every write is an explicit manual invocation."""

import argparse
import json
import sys
from .contracts import load_json


def main():
    parser = argparse.ArgumentParser(description="Fuxi v1 independent research platform; explicit manual campaigns and receipt-gated promotion")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("demo", "research"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--output", required=True, help="isolated new run root; never raw data")
        if name == "research":
            cmd.add_argument("--input", required=True, help="fuxi.input.1 JSON snapshot and explicit labels")
    check = commands.add_parser("verify")
    check.add_argument("run_dir")
    ingest = commands.add_parser("snapshot-local")
    ingest.add_argument("--raw-root", required=True)
    ingest.add_argument("--start", required=True)
    ingest.add_argument("--end", required=True)
    ingest.add_argument("--max-instruments", type=int, default=24)
    ingest.add_argument('--legacy-code-order',action='store_true',help='explicitly reproduce the old biased diagnostic sample')
    ingest.add_argument("--output", required=True)
    historical=commands.add_parser('snapshot-history',help='explicit cohort/window and supplemented event data')
    historical.add_argument('--plan',required=True)
    historical.add_argument('--raw-root',required=True)
    historical.add_argument('--supplement',required=True)
    historical.add_argument('--output',required=True)
    historical.add_argument('--remediation',help='explicit evidence config: vendor vintage and issuer adjudications')
    preflight=commands.add_parser('history-preflight',help='freeze and reconcile fixed account probes before expanded search')
    preflight.add_argument('--plan',required=True)
    preflight.add_argument('--input',required=True)
    preflight.add_argument('--output',required=True)
    study=commands.add_parser('history-study',help='freeze or resume historical reevaluation and memory controls')
    study.add_argument('action',choices=('freeze','run'))
    study.add_argument('--output',required=True)
    study.add_argument('--plan')
    study.add_argument('--candidates')
    study.add_argument('--input')
    history_export=commands.add_parser('history-export',help='publish a verified frozen result for operator review')
    history_export.add_argument('--run',required=True)
    history_export.add_argument('--formula',required=True)
    history_export.add_argument('--output',required=True)
    certify_data=commands.add_parser('certify-data',
        help='derive the data-contract flags from a frozen payload instead of asserting them')
    certify_data.add_argument('--input',required=True,help='frozen fuxi.history_input.1 payload')
    certify_data.add_argument('--plan',help='frozen plan supplying the cohort selection rule')
    certify_data.add_argument('--lifecycle',help='security master CSV; defaults to <raw>/delist_dates.csv')
    certify_data.add_argument('--raw-root',default='data/raw')
    certify_data.add_argument('--output',help='write the certificate here instead of printing it')
    certify_data.add_argument('--apply',action='store_true',
        help='also write the payload with the derived flags and market hash rebound')
    pit_cohort=commands.add_parser('pit-cohort',
        help='declare a point-in-time cohort and prove the rule cannot encode the outcome')
    pit_cohort.add_argument('--raw-root',default='data/raw')
    pit_cohort.add_argument('--start',required=True)
    pit_cohort.add_argument('--end',required=True)
    pit_cohort.add_argument('--salt',required=True)
    pit_cohort.add_argument('--per-exchange',type=int,default=48)
    pit_cohort.add_argument('--rule',choices=('per_exchange_hash_cutoff','full_population_hash_sample','all_a_share_census'),
                            default='per_exchange_hash_cutoff')
    pit_cohort.add_argument('--target-size',type=int)
    pit_cohort.add_argument('--output')
    system_demo = commands.add_parser("system-fixture")
    system_demo.add_argument("--output", required=True)
    campaign = commands.add_parser("campaign")
    campaign.add_argument("--input", required=True)
    campaign.add_argument("--output", required=True)
    campaign.add_argument("--rounds", type=int, default=5)
    campaign.add_argument("--candidates", type=int, default=3)
    campaign.add_argument("--mode", choices=("adaptive", "fixed", "no_memory"), default="adaptive")
    campaign.add_argument("--seed", type=int, default=17)
    campaign.add_argument("--llm", action="store_true")
    campaign.add_argument("--models", default="ridge,lightgbm,xgboost")
    campaign.add_argument("--resume-id")
    campaign.add_argument('--study-protocol')
    campaign.add_argument('--pause-after-round',type=int)
    campaign.add_argument('--data-receipt-id')
    campaign.add_argument('--scheduler-policy', choices=('legacy','selection_v2'), default='legacy')
    packages = commands.add_parser('work-packages', help='manual architecture study; never grants investment qualification')
    packages.add_argument('action', choices=('freeze','run','index','compare-parity','freeze-completion','run-completion','freeze-remediation','run-remediation','prepare-jq','return-jq','forward-plan','monitor'))
    packages.add_argument('--output', required=True)
    packages.add_argument('--input')
    packages.add_argument('--source')
    packages.add_argument('--styles')
    packages.add_argument('--llm', action='store_true')
    packages.add_argument('--status')
    packages.add_argument('--start')
    packages.add_argument('--end')
    packages.add_argument('--protocol-hash')
    packages.add_argument('--version')
    packages.add_argument('--candidate')
    packages.add_argument('--expected')
    packages.add_argument('--observed')
    packages.add_argument('--history')
    packages.add_argument('--request')
    packages.add_argument('--authority')
    packages.add_argument('--evidence', action='append', default=[])
    blueprint=commands.add_parser('blueprint',help='freeze, resume or inspect the complete source-bound research lifecycle')
    blueprint.add_argument('action',choices=('freeze','run','status'))
    blueprint.add_argument('--output',required=True)
    blueprint.add_argument('--request')
    archive=commands.add_parser('forward-archive',help='record actual prospective frozen decisions and native account pairs')
    archive.add_argument('action',choices=('freeze','capture','pair','observe','status'))
    archive.add_argument('--output',required=True)
    archive.add_argument('--request')
    archive.add_argument('--authority')
    from .operations import ACTIONS
    operation = commands.add_parser("operate")
    operation.add_argument("action",choices=ACTIONS)
    operation.add_argument("--request",required=True)
    operation.add_argument("--authority",required=True)
    args = parser.parse_args()
    try:
        if args.command == 'blueprint':
            from .blueprint_runner import freeze,run,status
            from .runtime import controlled_writer
            if args.action=='status':value=status(args.output)
            else:
                with controlled_writer(args.output):
                    if args.action=='freeze':
                        if not args.request:raise ValueError('--request required')
                        value=freeze(load_json(args.request),args.output)
                    else:value=run(args.output)
            body=value.get("complete") or value
            summary={k:body[k] for k in ("state","protocol_hash","cohort_size","engineering_execution_completed","restricted_scenarios","residual_scenarios","conditional_rights_scenarios","real_JQ_returns","real_forward_observations","new_qualifications","live_execution_allowed") if k in body}
            summary.update(output=args.output,completed_stages=value.get("completed_stages"))
            print(json.dumps(summary,ensure_ascii=False))
        elif args.command=='forward-archive':
            from .forward_archive import freeze_archive,capture_day,record_account_pair,build_observation,status
            from .runtime import controlled_writer
            from .evidence import EvidenceStore
            actions={'freeze':freeze_archive,'capture':capture_day,'pair':record_account_pair,'observe':build_observation}
            with controlled_writer(args.output):
                store=EvidenceStore(args.authority) if args.authority else None
                try:
                    if args.action=='status':value=status(args.output,store)
                    else:
                        if not args.request:raise ValueError('--request required')
                        spec=load_json(args.request)
                        value=freeze_archive(spec,args.output,store) if args.action=='freeze' else actions[args.action](args.output,spec,store)
                    print(json.dumps(value,ensure_ascii=False))
                finally:
                    if store is not None:store.close()
        elif args.command == 'work-packages':
            from .runtime import controlled_writer, authority_location
            from .architecture_study import freeze, run
            if args.action == 'index':
                from .factor_cards import query
                print(json.dumps(query(args.output, status=args.status, candidate=args.candidate, start=args.start, end=args.end, protocol_hash=args.protocol_hash, version=args.version), ensure_ascii=False))
            else:
                with controlled_writer(args.output):
                    if args.action == 'freeze':
                        if not args.input: raise ValueError('--input required')
                        value = freeze(args.input,args.output,source_path=args.source,styles_path=args.styles,llm=args.llm)
                        print(json.dumps({'protocol_hash':value['protocol_hash'],'output':args.output},ensure_ascii=False))
                    elif args.action in ('freeze-remediation','run-remediation'):
                        from .remaining_study import freeze as freeze_remediation, run as run_remediation
                        if args.action=='freeze-remediation':
                            if not args.input or not args.request:raise ValueError('--input and --request required')
                            value=freeze_remediation(args.input,args.output,load_json(args.request),industry=args.source,benchmark=args.observed)
                        else:value=run_remediation(args.output)
                        print(json.dumps({'protocol_hash':value['protocol_hash'],'result':value if args.action=='run-remediation' else 'frozen'},ensure_ascii=False))
                    elif args.action in ('freeze-completion','run-completion'):
                        from .architecture_completion import freeze as freeze_completion, run as run_completion
                        if args.action == 'freeze-completion':
                            if not args.history: raise ValueError('--history required')
                            value = freeze_completion(args.history,args.output,evidence_files=args.evidence,styles=args.styles)
                        else:
                            value = run_completion(args.output)
                        print(json.dumps({'protocol_hash':value['protocol_hash'],'result':value if args.action=='run-completion' else 'frozen'},ensure_ascii=False))
                    elif args.action in ('prepare-jq','return-jq','forward-plan','monitor'):
                        if not args.request: raise ValueError('--request required')
                        from .evidence import EvidenceStore
                        from .campaign import write_new
                        spec = load_json(args.request)
                        store = EvidenceStore(args.authority or authority_location(args.output))
                        try:
                            if args.action in ('prepare-jq','return-jq'):
                                from .manual_handoff import prepare, ingest
                                value = (prepare if args.action=='prepare-jq' else ingest)(store,spec)
                            elif args.action == 'forward-plan':
                                from .operational_monitor import forward_plan
                                value = forward_plan(spec)
                            else:
                                from .operational_monitor import monitor
                                value = monitor(spec,store)
                        finally:
                            store.close()
                        write_new(args.output,value)
                        print(json.dumps(value,ensure_ascii=False))
                    elif args.action == 'compare-parity':
                        if not args.expected or not args.observed: raise ValueError('expected and observed exports required')
                        from .diagnostic_parity import compare
                        from .campaign import write_new
                        value = compare(load_json(args.expected),load_json(args.observed))
                        write_new(args.output,value)
                        print(json.dumps(value,ensure_ascii=False))
                    else:
                        value = run(args.output,authority_path=authority_location(args.output))
                        print(json.dumps(value,ensure_ascii=False))
        elif args.command == "verify":
            from .artifacts import verify
            result = verify(args.run_dir)
            print(json.dumps({"verified": True, "run_id": result["run_id"],
                              "manifest_hash": result["manifest_hash"]}, ensure_ascii=False))
        elif args.command == "operate":
            from .operations import operate
            from .evidence import EvidenceStore
            from .runtime import controlled_writer
            with controlled_writer(args.authority):
                store = EvidenceStore(args.authority)
                try:
                    result = operate(store,args.action,load_json(args.request))
                    print(json.dumps(result,ensure_ascii=False))
                finally:
                    store.close()
        elif args.command=='history-export':
            from .historical_handoff import export_history
            from .runtime import controlled_writer
            with controlled_writer(args.output):
                print(json.dumps(export_history(args.run,args.formula,args.output),ensure_ascii=False))
        elif args.command=='certify-data':
            from pathlib import Path
            from .campaign import write_new
            from .certification import certify, derived_data_receipt
            from .pit_universe import read_lifecycle, read_lifecycle_file
            payload=load_json(args.input)
            plan=load_json(args.plan) if args.plan else None
            lifecycle=(read_lifecycle_file(args.lifecycle) if args.lifecycle
                       else read_lifecycle(args.raw_root))
            declaration={}
            if plan is not None:
                from .certification import cohort_declaration
                declaration={'cohort':cohort_declaration(plan,payload)}
            certificate=certify(payload,declaration=declaration,lifecycle=lifecycle)
            receipt=derived_data_receipt(certificate)
            if args.output:
                from .runtime import controlled_writer
                with controlled_writer(args.output):
                    write_new(args.output,certificate)
            if args.apply:
                from .certification import apply_to_payload
                if not args.output:raise ValueError('--apply requires --output for the written payload')
                write_new(str(Path(args.output).with_name(Path(args.output).stem+'_applied.json')),
                          apply_to_payload(payload,certificate))
            print(json.dumps({'certification_hash':certificate['certification_hash'],
                'snapshot_hash':certificate['snapshot_hash'],
                'flags':{k:v['value'] for k,v in certificate['flags'].items()},
                'blocking_gaps':certificate['blocking_gaps'],
                'residual_gaps':certificate['residual_gaps'],
                'claim_scope':certificate['claim_scope'],
                'receipt':{k:receipt[k] for k in ('pit_certified','universe_certified',
                    'account_events_certified','execution_rules_certified',
                    'market_representativeness_certified')}},ensure_ascii=False,indent=1))
        elif args.command=='pit-cohort':
            from pathlib import Path
            from .campaign import write_new
            from .datasets import read_calendar
            from .pit_universe import build_pit_cohort, read_lifecycle, reproduce_cohort
            import hashlib
            lifecycle=read_lifecycle(args.raw_root)
            sessions,_=read_calendar(Path(args.raw_root)/'trade_calendar.csv',args.start,args.end)
            if args.rule=='full_population_hash_sample':
                if not args.target_size:raise ValueError('--target-size required for a whole-population sample')
                declaration={'rule_kind':args.rule,'salt':args.salt,'target_size':args.target_size,
                             'sampling_key_fields':['ts_code'],'sampling_frame':'full_lifecycle_table',
                             'posthoc_exclusion':False,'rule':'whole-population identifier-keyed hash sample'}
                codes,basis=reproduce_cohort(lifecycle,declaration,sessions)
            else:
                codes,declaration,basis=build_pit_cohort(lifecycle,sessions,salt=args.salt,
                                                         per_exchange=args.per_exchange,rule_kind=args.rule)
            if codes is None:raise ValueError('cohort rule not evaluable: '+str(basis.get('blocking_gaps')))
            from .pit_universe import neutrality_audit
            body={'declaration':declaration,'codes':codes,'reproduction':basis,
                  'neutrality':neutrality_audit(lifecycle,sessions,codes,all_a=args.rule=='all_a_share_census'),
                  'cohort_hash':hashlib.sha256(json.dumps(codes).encode()).hexdigest()}
            if args.output:
                from .runtime import controlled_writer
                with controlled_writer(args.output):
                    write_new(args.output,body)
            print(json.dumps({'cohort_size':len(codes),'rule_kind':declaration.get('rule_kind'),
                'delisting_names_in_panel':body['neutrality']['cohort_names_delisting_inside_window'],
                'population_delisting_names':body['neutrality']['population_names_delisting_inside_window'],
                'expected_if_neutral':body['neutrality']['expected_cohort_delisting_count'],
                'survivor_filter_detected':body['neutrality']['survivor_filter_detected']},
                ensure_ascii=False))
        elif args.command in {'snapshot-history','history-study','history-preflight'}:
            from pathlib import Path
            from .runtime import controlled_writer
            from .campaign import write_new
            with controlled_writer(args.output):
                if args.command=='snapshot-history':
                    from .historical_adapter import build_history_input
                    result=build_history_input(args.raw_root,load_json(args.plan),args.supplement,Path(args.output).parent/(Path(args.output).stem+'_audit'),
                                              remediation=load_json(args.remediation) if args.remediation else None)
                    write_new(args.output,result)
                    print(json.dumps(result['audit'],ensure_ascii=False))
                elif args.command=='history-preflight':
                    from .remediation import preflight
                    result=preflight(args.input,args.plan,args.output)
                    print(json.dumps({k:result[k] for k in ['probe_count','search_allowed','formal_account_evidence_certified']},ensure_ascii=False))
                else:
                    from .historical_study import freeze,run
                    if args.action=='freeze':
                        if not all((args.plan,args.candidates,args.input)):raise ValueError('freeze requires --plan --candidates --input')
                        result=freeze(args.plan,args.candidates,args.input,args.output)
                    else:result=run(args.output, args.data_receipt_id)
                    print(json.dumps(result,ensure_ascii=False))
        elif args.command in {"snapshot-local", "system-fixture", "campaign"}:
            from .runtime import controlled_writer, authority_location
            from .campaign import run_campaign, write_new
            with controlled_writer(args.output):
                if args.command == "snapshot-local":
                    if not args.legacy_code_order:
                        raise ValueError('Use snapshot-history with an explicit plan; old code-order sampling requires --legacy-code-order for reproduction only')
                    from .datasets import local_market_input
                    result = local_market_input(args.raw_root, args.start, args.end, args.max_instruments)
                    write_new(args.output, result)
                    print(json.dumps({"snapshot_path": args.output, "sessions": len(result["snapshot"]["sessions"]),
                                      "instruments": len(result["snapshot"]["universe"][result["snapshot"]["sessions"][0]]),
                                      "quality": result["snapshot"]["quality"]}, ensure_ascii=False))
                elif args.command == "system-fixture":
                    from .system_fixture import synthetic_campaign
                    write_new(args.output, synthetic_campaign())
                    print(json.dumps({"synthetic_fixture": args.output}))
                else:
                    path, report = run_campaign(args.input, args.output, rounds=args.rounds,
                         candidates_per_round=args.candidates, mode=args.mode, seed=args.seed, llm_enabled=args.llm,
                         model_algorithms=tuple(x for x in args.models.split(",") if x), campaign_id=args.resume_id,
                         authority_path=authority_location(args.output),
                         study_protocol=load_json(args.study_protocol) if args.study_protocol else None,
                         pause_after_round=args.pause_after_round,data_receipt_id=args.data_receipt_id, scheduler_policy=args.scheduler_policy)
                    print(json.dumps({"report_path": str(path), "campaign_id": report["campaign_id"],
                        "rounds_completed": len(report["rounds"]), "attempted_candidates": report["attempted_candidates"],
                        "engineering_errors": report["engineering_errors"], "jq_candidate_count": report["jq_candidate_count"]}, ensure_ascii=False))
        else:
            from .pipeline import run
            from .fixtures import synthetic_input
            from .runtime import controlled_writer
            payload = synthetic_input() if args.command == "demo" else load_json(args.input)
            with controlled_writer(args.output):
                path, summary = run(payload, args.output)
            print(json.dumps({"run_dir": str(path.resolve()), "summary": summary}, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
        print(json.dumps({"eval_status": "error", "decision": "not_applicable",
                          "error_type": type(exc).__name__, "reason": str(exc),
                          "jq_candidate_allowed": False}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
