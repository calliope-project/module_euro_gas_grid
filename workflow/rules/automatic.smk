"""Rules to used to download automatic resource files."""

rule user_input_shapes:
    message:
        "Download spatial zones."
    params:
        url=internal["resources"]["automatic"]["spatial_units"],
    output:
        "resources/user/shapes_national.geojson",
    conda:
        "../envs/shell.yaml"
    localrule: True
    shell:
        "curl -sSLo {output} '{params.url}'"

rule download_sci_grid_data:
    message: "Download gas infrastructure data from SciGRID_gas IGGIELGN"
    params:
        url = internal["resources"]["automatic"]["SciGRID_gas"]
    output: "resources/automatic/gas_grid.zip"
    conda: "../envs/shell.yaml"
    shell:
        """curl -sSLo {output} {params.url}"""

