"""python -m factor_mining 入口，转发到 cli.main。"""
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
