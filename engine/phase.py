import argparse
import engine
import engine.run_ralphi as run


if __name__ == "__main__":
    print("***********************************************")
    print("*  ralphi (%s): haplotype phasing mode *" % engine.__version__)
    print("***********************************************")
    parser = argparse.ArgumentParser(description='Haplotype assembly')
    parser.add_argument('--config', help='YAML config file')
    args = parser.parse_args()

    run.initialize(args, False)
