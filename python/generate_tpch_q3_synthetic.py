#!/usr/bin/env python3
"""Compatibility shim re-exporting canonical generator from lib."""
from lib.generate_tpch_q3_synthetic import *

if __name__ == "__main__":
    from lib.generate_tpch_q3_synthetic import main
    main()
