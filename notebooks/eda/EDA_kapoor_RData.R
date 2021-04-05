## Set up environment
# install.packages("flashClust")
library(WGCNA)
library(flashClust)
if (.Platform$OS.type == "windows") {prefix = 'G:'} else {prefix = '/Volumes/GoogleDrive'}
setwd(file.path(prefix, 'Shared drives/NIAAA_ASSIST/Data/'))
# Load Kapoor expression data
################################################################################################
## Getting the expression file from the kapoor's RData
# load('coga-inia.wgcna.hub.genes.files.RData')
# write.table(datExprOut, 'kapoor_expression_Apr5.txt', sep = '\t')

## Export Kapoor's module assignment
# gene_ids = colnames(datExprOut)
# wgcna_modules = data.frame(id = gene_ids, cluster = moduleColors)
# write.csv(wgcna_modules, './eda_derived/kapoor_wgcna_modules.csv', row.names = F)
# 
# ## Export Kapoor's TOM file
# colnames(TOM) = colnames(adjacency)
# rownames(TOM) = rownames(adjacency)
# 
# write.csv(TOM, 'kapoor_TOM_Apr5.txt')
################################################################################################

# Load Kapoor expression data
datExpr = read.delim('kapoor_expression_Apr5.txt')

# Calculate the adjacencies.
adjacency = adjacency(datExpr, power = 14, type='signed')
# Turn adjacency into topological overlap
TOM = TOMsimilarity(adjacency, TOMType = "signed")
# Calculate the corresponding dissimilarity.
dissTOM <- 1-TOM
# Call the hierarchical clustering function.
geneTree = flashClust(as.dist(dissTOM), method = "average")
# Set the minimum module size.
minModuleSize = 100
# Set the cutting height.
detectCutHeight = 0.99
# Module identification using dynamic tree cut.
dynamicMods = cutreeDynamic(dendro = geneTree, cutHeight=detectCutHeight, deepSplit = T, minClusterSize = minModuleSize);
# Display module size for each module.
table(dynamicMods)
# Convert numeric lables into colors.
dynamicColors = labels2colors(dynamicMods)
table(dynamicColors)
print(length(table(dynamicColors)))
# Plot the dendrogram and colors underneath.
pdf(file=paste0(directory,"network_dendrogram.pdf"), width = 14, height = 7)# Set graphical parameters.
plotDendroAndColors(dendro=geneTree, colors=dynamicColors, groupLabels="35.99T", rowText=dynamicColors, cex.rowText = 0.5, dendroLabels = F, hang = 0.03, addGuide = TRUE, guideHang = 0.05, main = paste0("Gene dendrogram and module colors"))


# Call an automatic merging function
merge = mergeCloseModules(datExpr, dynamicColors, cutHeight = 0.1, verbose = 3)
# The merged module colors
length(table(merge$colors));
head(merge$colors)

save(datExpr, adjacency, TOM, dissTOM, geneTree, minModuleSize, detectCutHeight, 
     dynamicMods, dynamicColors, merge, file = 'Kapoor_network_analysis.RData')

