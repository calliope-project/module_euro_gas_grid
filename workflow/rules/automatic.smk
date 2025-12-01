"""Rules to used to download automatic resource files."""

rule download_sci_grid:
    message: "Download gas infrastructure data from SciGRID_gas IGGIELGN"
    params:
        url = internal["resources"]["automatic"]["scigrid_gas"]
    log:
        "logs/download_sci_grid.log",
    output:
        zipfile=temp("resources/automatic/gas_grid.zip")
    conda:
        "../envs/shell.yaml"
    shell:
        """curl -sSLo {output} {params.url}"""

rule unzip_pipe_segements:
    message:
        "Unzipping SciGrid '{params.file}'."
    params:
        file=f"data/IGGIELGN_PipeSegments.geojson",
    input:
        zip_file=rules.download_sci_grid.output.zipfile,
    output:
        pipelines=temp("resources/automatic/pipesegments.geojson"),
    log:
        "logs/automatic/unzip_pipe_segements.log",
    conda:
        "../envs/clustering.yaml"
    script:
        "../scripts/unzip.py"
