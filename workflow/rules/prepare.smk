"""Rules to standardise files."""


rule prepare_countries:
    message:
        "Preparing country data."
    input:
        raw_countries="resources/automatic/countries.zip",
    output:
        countries="resources/automatic/countries.parquet",
        fig="resources/automatic/countries.png"
    log:
        "logs/prepare_countries.log",
    conda:
        "../envs/euro_gas_grid.yaml"
    script:
        "../scripts/prepare_countries.py"



rule prepare_pipelines:
    message:
        "Harmonising SciGRID pipelines."
    params:
        imputation=config.get("imputation", {}),
        projected_crs = config["crs"]["projected"],
    input:
        raw_pipelines="resources/automatic/scigrid_gas/PipeSegments.geojson",
        raw_nodes="resources/automatic/scigrid_gas/Nodes.geojson",
        countries=rules.prepare_countries.output.countries
    output:
        pipelines="resources/automatic/pipelines.parquet",
        nodes="resources/automatic/nodes.parquet",
        fig="resources/automatic/pipelines.png",
    log:
        "logs/prepare_pipelines.log",
    conda:
        "../envs/euro_gas_grid.yaml"
    script:
        "../scripts/prepare_pipelines.py"
