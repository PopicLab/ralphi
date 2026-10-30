import argparse
import engine
import engine.run_ralphi as run
import engine.config as config_utils


if __name__ == "__main__":
    print("***********************************************")
    print("*  ralphi (%s): haplotype linkage mode *" % engine.__version__)
    print("***********************************************")
    parser = argparse.ArgumentParser(description='Haplotype assembly')
    parser.add_argument('--config', help='YAML config file')
    args = parser.parse_args()
    config_type = config_utils.CONFIG_TYPE.LINK
    config = config_utils.load_config(args.config, config_type=config_type)
    run.initialize(config)
