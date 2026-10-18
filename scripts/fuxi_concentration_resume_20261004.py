"""Create the required output directory before invoking the immutable frozen runner."""
import argparse,sys,subprocess,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fuxi_v1.contracts import load_json,require
from fuxi_v1.datasets import file_hash
from fuxi_v1.runtime import controlled_writer
OUT=ROOT/'reports/fuxi_v1_concentration_20261004'


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True);p.add_argument('--rules');p.add_argument('--exclude',nargs='*',default=[])
    args=p.parse_args();require(args.phase.isidentifier(),'phase must be one directory name')
    runner=ROOT/'scripts/fuxi_concentration_20261004.py'
    with controlled_writer(OUT):
        require(file_hash(runner)==load_json(OUT/'PLAN.json')['runner_sha256'],'frozen runner changed')
        directory=OUT/args.phase;directory.mkdir(parents=True,exist_ok=True);(directory/'accounts').mkdir(exist_ok=True)
        receipt=directory/'OUTPUT_PRECONDITION.json'
        if not receipt.exists():
            receipt.write_text(json.dumps({'accounts_directory_created':True,'frozen_runner_unchanged':True,
                'launcher_sha256':file_hash(Path(__file__))},indent=2),encoding='utf-8')
    command=[sys.executable,'-X','utf8',str(runner),'run','--phase',args.phase]
    if args.rules:command.extend(['--rules',args.rules])
    if args.exclude:command.extend(['--exclude',*args.exclude])
    sys.exit(subprocess.run(command,cwd=ROOT).returncode)


if __name__=='__main__':main()
