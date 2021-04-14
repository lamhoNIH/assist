# BioC / WGCNA pull in latest RQSLite, which fails to build because of an incomplete Boost lib.
# Try to pull in an earlier version before installing WGCNA -- https://stackoverflow.com/a/29840882
install.packages("remotes")
require(remotes)
install_version("RSQLite", version = "2.2.4", repos = "http://cran.us.r-project.org")
library(RSQLite)
install.packages("rjson")
install.packages("flashClust", version='1.1.25')
install.packages("BiocManager", version='3.11')
library(BiocManager)
BiocManager::install("WGCNA")