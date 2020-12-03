# Script to generate WCGNA network
# Set up the environment
install.packages("BiocManager")
BiocManager::install("WGCNA")
library(WGCNA)

## automatic block-wise network
# load expression data
expression = read.csv('G:/Shared drives/NIAAA_ASSIST/Data/network_only_expression.csv', 
                      header = T, row.names = 1)
# transpose the dataframe
expression_t = t(expression)

# blockwiseModules() will generate Tom network along with the module detection by WCGNA
net = blockwiseModules(expression_t, power = 14,
                       TOMType = "unsigned", minModuleSize = 100,
                       reassignThreshold = 0, detectCutHeight = 0.99,
                       numericLabels = TRUE, pamRespectsDendro = FALSE,
                       saveTOMs = F,
                       #   saveTOMFileBase = "femaleMouseTOM",
                       verbose = 3, deepSplit = T)

# summary of the network modules (colors represent module assignment)
table(net$colors)
# output the module assignment
write.csv(net$colors, 'G:/Shared drives/NIAAA_ASSIST/Data/eda_derived/wcgna_modules.csv')