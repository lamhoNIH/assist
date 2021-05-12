This R markdown file shows the codes for the ‘Network analysis’ module
================

``` r
knitr::opts_chunk$set(echo = TRUE)
```

    ## Warning: package 'WGCNA' was built under R version 4.0.5

    ## Loading required package: dynamicTreeCut

    ## Loading required package: fastcluster

    ## 
    ## Attaching package: 'fastcluster'

    ## The following object is masked from 'package:stats':
    ## 
    ##     hclust

    ## 

    ## 
    ## Attaching package: 'WGCNA'

    ## The following object is masked from 'package:stats':
    ## 
    ##     cor

``` r
if (Sys.info()["sysname"] == "Windows") {prefix = "G:"} else {prefix = "/Volumes/GoogleDrive"}
expression = read.table(paste0(prefix,'/Shared drives/NIAAA_ASSIST/Data/kapoor_expression.txt'), header = TRUE)
rownames(expression) = expression$id
expression$id = NULL
expression_t = t(expression)
print(expression_t[1:5,1:5])
```

    ##      ENSG00000227232 ENSG00000237683 ENSG00000241860 ENSG00000228463
    ## X214        3.133118        3.823457      1.02268961        4.174664
    ## X460        2.389945        1.997970      0.36883000        3.233721
    ## X584        1.877375        2.878354      0.02826542        3.250095
    ## X551        2.657129        1.632495      0.24708347        4.155158
    ## X530        3.186562        3.185500      1.10050055        2.999840
    ##      ENSG00000225972
    ## X214        4.721111
    ## X460        5.096691
    ## X584        4.613348
    ## X551        5.912895
    ## X530        5.616984
