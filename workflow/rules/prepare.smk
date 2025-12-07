"""Rules to standardise files."""

rule prepare_shapes:
    message:
        "Preparing shapes for the Euro Gas Grid module."
    input:
        shapes="resources/user/{shapes}/shapes.parquet",
        north_sea=rules.download_north_sea.output.zipfile if config["add_north_sea"] else None
    output:
        shapes="resources/automatic/{shapes}/shapes.parquet",
        fig="resources/automatic/{shapes}/shapes.png",
    log:
        "logs/{shapes}/prepare_shapes.log"
    conda:
        "../envs/euro_gas_grid.yaml"
    script:
        "../scripts/prepare_shapes.py"

rule prepare_landmass:
    message:
        "Preparing landmass data."
    input:
        raw_landmass="resources/automatic/landmass.zip",
    output:
        landmass="resources/automatic/landmass.parquet",
        fig="resources/automatic/landmass.png"
    log:
        "logs/prepare_landmass.log",
    conda:
        "../envs/euro_gas_grid.yaml"
    script:
        "../scripts/prepare_landmass.py"

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
        "Harmonising SciGRID data."
    params:
        imputation=config["imputation"],
    input:
        raw_pipelines="resources/automatic/scigrid_gas/PipeSegments.geojson",
        raw_nodes="resources/automatic/scigrid_gas/Nodes.geojson",
        landmass=rules.prepare_landmass.output.landmass,
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
