import argparse
import engine
import engine.run_ralphi as run
import engine.config as config_utils


if __name__ == "__main__":
    print("***********************************************")
    print("*  ralphi (%s): haplotype phasing mode *" % engine.__version__)
    print("***********************************************")
    parser = argparse.ArgumentParser(description='Haplotype assembly')
    parser.add_argument('--config', help='YAML config file')
    args = parser.parse_args()

    config_type = config_utils.CONFIG_TYPE.TEST
    config = config_utils.load_config(args.config, config_type=config_type)

    run.initialize(config)
    if any(inp.link_bam for inp in config.inputs):
        # Hierarchical phasing and then linkage
        config.to_link()
        run.initialize(config)
    