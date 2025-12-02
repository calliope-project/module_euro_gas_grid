"""Rules to standardise files."""


rule prepare_landmass:
    message:
        "Preparing landmass data."
    input:
        raw_folder="resources/automatic/landmass/",
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
        raw_folder="resources/automatic/countries/",
        raw_pipelines=rules.unzip_pipe_segements.output.pipelines,
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
        "Validating SciGRID data."
    params:
        projected_crs=config["crs"]["projected"],
        imputation=config["imputation"],
    input:
        raw_pipelines=rules.unzip_pipe_segements.output.pipelines,
        landmass=rules.prepare_landmass.output.landmass
    output:
        pipelines="resources/automatic/pipelines.parquet",
        fig_offshore="resources/automatic/pipelines_offshore.png",
        fig_ch4_capacity="resources/automatic/pipelines_ch4_capacity.png"
    log:
        "logs/prepare_pipelines.log",
    conda:
        "../envs/euro_gas_grid.yaml"
    script:
        "../scripts/prepare_pipelines.py"
