#load and read wine data with labels
library(readxl)
wine <- read_excel("wine.xlsx")
label = wine[1]
data=wine[2:14]
#pre-process the data (standarzitaion)
data.s=scale(data,center = TRUE,scale = TRUE)
#PCA analysis
data.cor=cor(data)
data.eig=eigen(data.cor)#gives us eigen values and vectors
data.eigen=cbind(data.eig$values)#eigen values assigened
#plot of eiegenvalues
plot(data.eigen,ylim=c(0,5),pch=20,main = "Scree Plot for PCs",ylab = "eigenvalues/pcvalues")
par(bg="gray")
grid(nx=NULL,col = "yellow")
#process to plot cumulative sum in percentage
Prdata.eignen=data.eigen/sum(data.eigen)
percentage=100*cumsum(Prdata.eignen)
plot(percentage,pch=20,main = "plot of cumulative percentage assess for eigenvalues",ylab = "cumulative sum in percentage")
par(bg="gray")
grid(nx=NULL,col = "yellow")
#projection of the data using principal componenents(eigenvalues)
data.new=data.s%*%data.eig$vectors
#projection of the data with first 4 PCs which decided from cumsum
#because first 4 PCs give %73.59
data.RD=data.new[,c(1:4)]
#hierarchical cluster analysis
d=dist(data.s, method = "euclidean")
dataclust=hclust(d,method = "ward.D2")
#plot dendogram
plot(dataclust,cex=0.40,labels=t(label),hang = -0.1)
par(bg="gray")
rect.hclust(dataclust, k=3, border="red")
#rect.hclust(dataclust, k=5, border="red") 


