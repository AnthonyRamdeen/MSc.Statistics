library(stats)
library(MASS)
library(lattice)
library(mvtnorm)
library(gmm)
library(tmvtnorm)
library(cluster)
library(ForImp)
library(pcaPP)
library(robustbase)
library(rrcov)
library(e1071)


#Loading Image Data and Keypoint Data
data=read.csv("training.csv", stringsAsFactors=F)
data=as.matrix(data)
attach(data)
str(data)
head(data)
pixels=data$Image
data$Image=NULL

#Converting the Image strings to Dataframe
image.data=matrix(nrow=7049,ncol=(96*96))

for(i in 1:7049){
image.data[i,]=as.vector(rev(as.integer(unlist(strsplit(pixels[i], " ")))))
}
image.data=as.data.frame(image.data)
str(image.data)



#Check the missingness of the data by columns
na.rep999.data=na.replace(as.matrix(data), 999)

missing.vec=vector(length=30)
for(i in 1:30){

missing.vec[i]=length(which((((na.rep999.data[,i])==999))))
}
missing.mat=matrix(missing.vec,ncol=2,byrow=T)
missing.mat=as.data.frame(missing.mat)
missing.mat[,1]-missing.mat[,2]

missing.indexes=as.vector(missing.mat[,1])

check.index.vec=vector(length=15)
for(i in 1:15){
check.index.vec[i]=+sum(which( (is.na(data[,((2*i)-1)])==T))
	- which( (is.na(data[,(2*i)])==T)))
}
check.index.vec


#Creating the image from a Pixel Intensity Matrix
#1278 - perfect image used
#the rev makes the picture upright
pic1=matrix((as.numeric(image.data[1278,])),96,96) 
pic=image(1:96, 1:96, (pic1), col=gray((0:255)/255),axes=F,xlab="",ylab="")


#Plotting keypoints for any given image
x.indexes=seq(1,29,2)
y.indexes=seq(2,30,2)
x.key=vector(length=15)
y.key=vector(length=15)
x.key=as.vector(data[1278,x.indexes]) 
y.key=as.vector(data[1278,y.indexes]) 
points(96-x.key, 96-y.key,pch=21,col=4,bg=3,cex=2)


#Investigating the average keypoint values on an image
mean.x.key=colMeans(data[,x.indexes],na.rm=T)
mean.y.key=colMeans(data[,y.indexes],na.rm=T)
points(mean.x.key, mean.y.key,col="red")
mean.pair.key=cbind(mean.x.key,mean.y.key)
rownames(mean.pair.key)<-NULL
mean.pair.key
abline(v=48,h=48,col="orange")


par(mfrow=c(1,2))



#Case deleted dataset
cd.data=na.omit(cbind(data,image.data))

#Case Deleted Dataset Split indices 
#Also used for the ROBPCA to ensure comparativeness
#These were run once and only once as 
#Any subsequent running will change the indices
cd.sample1 <- 
sample.int(n = nrow(cd.data), size = floor(.50*nrow(cd.data)), replace = F)
cd.sample2 <- 
sample.int(n = nrow(cd.data), size = floor(.50*nrow(cd.data)), replace = F)
cd.sample3 <- 
sample.int(n = nrow(cd.data), size = floor(.50*nrow(cd.data)), replace = F)
cd.sample4 <- 
sample.int(n = nrow(cd.data), size = floor(.50*nrow(cd.data)), replace = F)
cd.sample5 <- 
sample.int(n = nrow(cd.data), size = floor(.50*nrow(cd.data)), replace = F)

#Ensured that the indexed rows stayed the same
cd.sample1 <-cd.sample1
cd.sample2 <-cd.sample2
cd.sample3 <-cd.sample3
cd.sample4 <-cd.sample4
cd.sample5 <-cd.sample5

#Breaking down the data in their respective folds and interations

###First fold, first iteration
#Assess components by Scree-plot and Eigenvalues >=1
ptm <- proc.time()
cd.cpca.f1.it1=
prcomp(cd.data[cd.sample1,31:9246],center=T,scale=T)
proc.time() - ptm
summary(cd.cpca.f1.it1)

screeplot(cd.cpca.f1.it1, type="l",npcs=50)
abline(h=0,col="blue")
length(which((cd.cpca.f1.it1$sdev)^2>=1))

# 324 PC's under eigenvalues
# 30 PC's under Scree-plot

#Extracting loadings and score matrices
cd.cpca.f1.it1.loading324=cd.cpca.f1.it1$rotation[,1:324]
cd.cpca.f1.it1.loading30=cd.cpca.f1.it1$rotation[,1:30]
cd.cpca.f1.it1.score324=cd.cpca.f1.it1$x[,1:324]
cd.cpca.f1.it1.score30=cd.cpca.f1.it1$x[,1:30]

#Transforming test data into pca-scores
cd.cpca.f1.it1.test.score324=
as.matrix(scale(cd.data[-cd.sample1,31:9246],center=T,scale=T))%*%
cd.cpca.f1.it1.loading324
cd.cpca.f1.it1.test.score30=
as.matrix(scale(cd.data[-cd.sample1,31:9246],center=T,scale=T))%*%
cd.cpca.f1.it1.loading30

#Support Vector Regression Models with overal RMSE Estimates
cd.cpca.f1.it1.324.rmse.vec=vector(length=30)

for (i in 1:30){
output<-cd.data[cd.sample1,i]
model.data<-as.data.frame(cbind(output,cd.cpca.f1.it1.score324))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,cd.cpca.f1.it1.test.score324)
cd.cpca.f1.it1.324.rmse.vec[i]<-
sqrt(mean((svr.pred-(cd.data[-cd.sample1,i]))^2))
}

sqrt(mean(rowMeans
	(matrix(cd.cpca.f1.it1.324.rmse.vec,ncol=2,byrow=T))^2))

cd.cpca.f1.it1.30.rmse.vec=vector(length=30)
for (i in 1:30){
output<-cd.data[cd.sample1,i]
model.data<-as.data.frame(cbind(output,cd.cpca.f1.it1.score30))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,cd.cpca.f1.it1.test.score30)
cd.cpca.f1.it1.30.rmse.vec[i]<-
sqrt(mean((svr.pred-(cd.data[-cd.sample1,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(cd.cpca.f1.it1.30.rmse.vec,ncol=2,byrow=T))^2))



###Second fold, first iteration
ptm <- proc.time()
cd.cpca.f2.it1=
prcomp(cd.data[-cd.sample1,31:9246],center=T,scale=T)
proc.time() - ptm
summary(cd.cpca.f2.it1)

screeplot(cd.cpca.f2.it1, type="l",npcs=50)
abline(h=0,col="blue")
length(which((cd.cpca.f2.it1$sdev)^2>=1))

# 325 PC's under eigenvalues
# 33 PC's under Scree-plot

#Extracting loadings and score matrices
cd.cpca.f2.it1.loading325=cd.cpca.f2.it1$rotation[,1:325]
cd.cpca.f2.it1.loading33=cd.cpca.f2.it1$rotation[,1:33]
cd.cpca.f2.it1.score325=cd.cpca.f2.it1$x[,1:325]
cd.cpca.f2.it1.score33=cd.cpca.f2.it1$x[,1:33]

#Transforming test data into pca-scores
cd.cpca.f2.it1.test.score325=
as.matrix(scale(cd.data[cd.sample1,31:9246],center=T,scale=T))%*%
cd.cpca.f2.it1.loading325
cd.cpca.f2.it1.test.score33=
as.matrix(scale(cd.data[cd.sample1,31:9246],center=T,scale=T))%*%
cd.cpca.f2.it1.loading33

#Support Vector Regression Models with overal RMSE Estimates
cd.cpca.f2.it1.325.rmse.vec=vector(length=30)

for (i in 1:30){
output<-cd.data[-cd.sample1,i]
model.data<-as.data.frame(cbind(output,cd.cpca.f2.it1.score325))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,cd.cpca.f2.it1.test.score325)
cd.cpca.f2.it1.325.rmse.vec[i]<-
sqrt(mean((svr.pred-(cd.data[cd.sample1,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(cd.cpca.f2.it1.325.rmse.vec,ncol=2,byrow=T))^2))

cd.cpca.f2.it1.33.rmse.vec=vector(length=30)
for (i in 1:30){
output<-cd.data[-cd.sample1,i]
model.data<-as.data.frame(cbind(output,cd.cpca.f2.it1.score33))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,cd.cpca.f2.it1.test.score33)
cd.cpca.f2.it1.33.rmse.vec[i]<-
sqrt(mean((svr.pred-(cd.data[cd.sample1,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(cd.cpca.f2.it1.33.rmse.vec,ncol=2,byrow=T))^2))



##First Fold, Second Iteration
ptm <- proc.time()
cd.cpca.f1.it2=
prcomp(cd.data[cd.sample2,31:9246],center=T,scale=T)
proc.time() - ptm
summary(cd.cpca.f1.it2)

screeplot(cd.cpca.f1.it2, type="l",npcs=50)
abline(h=0,col="blue")
length(which((cd.cpca.f1.it2$sdev)^2>=1))

# 326 PC's under eigenvalues
# 33  PC's under Scree-plot

#Extracting loadings and score matrices
cd.cpca.f1.it2.loading326=cd.cpca.f1.it2$rotation[,1:326]
cd.cpca.f1.it2.loading33=cd.cpca.f1.it2$rotation[,1:33]
cd.cpca.f1.it2.score326=cd.cpca.f1.it2$x[,1:326]
cd.cpca.f1.it2.score33=cd.cpca.f1.it2$x[,1:33]

#Transforming test data into pca-scores
cd.cpca.f1.it2.test.score326=
as.matrix(scale(cd.data[-cd.sample2,31:9246],center=T,scale=T))%*%
cd.cpca.f1.it2.loading326
cd.cpca.f1.it2.test.score33=
as.matrix(scale(cd.data[-cd.sample2,31:9246],center=T,scale=T))%*%
cd.cpca.f1.it2.loading33

#Support Vector Regression Models with overal RMSE Estimates
cd.cpca.f1.it2.326.rmse.vec=vector(length=30)

for (i in 1:30){
output<-cd.data[cd.sample2,i]
model.data<-as.data.frame(cbind(output,cd.cpca.f1.it2.score326))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,cd.cpca.f1.it2.test.score326)
cd.cpca.f1.it2.326.rmse.vec[i]<-
sqrt(mean((svr.pred-(cd.data[-cd.sample2,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(cd.cpca.f1.it2.326.rmse.vec,ncol=2,byrow=T))^2))

cd.cpca.f1.it2.33.rmse.vec=vector(length=30)
for (i in 1:30){
output<-cd.data[cd.sample2,i]
model.data<-as.data.frame(cbind(output,cd.cpca.f1.it2.score33))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,cd.cpca.f1.it2.test.score33)
cd.cpca.f1.it2.33.rmse.vec[i]<-
sqrt(mean((svr.pred-(cd.data[-cd.sample2,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(cd.cpca.f1.it2.33.rmse.vec,ncol=2,byrow=T))^2))



##Second Fold, Second Iteration
ptm <- proc.time()
cd.cpca.f2.it2=
prcomp(cd.data[-cd.sample2,31:9246],center=T,scale=T)
proc.time() - ptm
summary(cd.cpca.f2.it2)

screeplot(cd.cpca.f2.it2, type="l",npcs=50)
abline(h=0,col="blue")
length(which((cd.cpca.f2.it2$sdev)^2>=1))

# 323 PC's under eigenvalues
# 30 PC's under Scree-plot

#Extracting loadings and score matrices
cd.cpca.f2.it2.loading323=cd.cpca.f2.it2$rotation[,1:323]
cd.cpca.f2.it2.loading30=cd.cpca.f2.it2$rotation[,1:30]
cd.cpca.f2.it2.score323=cd.cpca.f2.it2$x[,1:323]
cd.cpca.f2.it2.score30=cd.cpca.f2.it2$x[,1:30]

#Transforming test data into pca-scores
cd.cpca.f2.it2.test.score323=
as.matrix(scale(cd.data[cd.sample2,31:9246],center=T,scale=T))
%*%cd.cpca.f2.it2.loading323
cd.cpca.f2.it2.test.score30=
as.matrix(scale(cd.data[cd.sample2,31:9246],center=T,scale=T))
%*%cd.cpca.f2.it2.loading30

#Support Vector Regression Models with overal RMSE Estimates
cd.cpca.f2.it2.323.rmse.vec=vector(length=30)

for (i in 1:30){
output<-cd.data[-cd.sample2,i]
model.data<-as.data.frame(cbind(output,cd.cpca.f2.it2.score323))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,cd.cpca.f2.it2.test.score323)
cd.cpca.f2.it2.323.rmse.vec[i]<-
sqrt(mean((svr.pred-(cd.data[cd.sample2,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(cd.cpca.f2.it2.323.rmse.vec,ncol=2,byrow=T))^2))

cd.cpca.f2.it2.30.rmse.vec=vector(length=30)
for (i in 1:30){
output<-cd.data[-cd.sample2,i]
model.data<-as.data.frame(cbind(output,cd.cpca.f2.it2.score30))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,cd.cpca.f2.it2.test.score30)
cd.cpca.f2.it2.30.rmse.vec[i]<-
sqrt(mean((svr.pred-(cd.data[cd.sample2,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(cd.cpca.f2.it2.30.rmse.vec,ncol=2,byrow=T))^2))



#Fold One, Iteration 3
ptm <- proc.time()
cd.cpca.f1.it3=
prcomp(cd.data[cd.sample3,31:9246],center=T,scale=T)
proc.time() - ptm
summary(cd.cpca.f1.it3)

screeplot(cd.cpca.f1.it3, type="l",npcs=50)
abline(h=0,col="blue")
length(which((cd.cpca.f1.it3$sdev)^2>=1))

# 325 PC's under eigenvalues
# 34 PC's under Scree-plot

#Extracting loadings and score matrices
cd.cpca.f1.it3.loading325=cd.cpca.f1.it3$rotation[,1:325]
cd.cpca.f1.it3.loading34=cd.cpca.f1.it3$rotation[,1:34]
cd.cpca.f1.it3.score325=cd.cpca.f1.it3$x[,1:325]
cd.cpca.f1.it3.score34=cd.cpca.f1.it3$x[,1:34]

#Transforming test data into pca-scores
cd.cpca.f1.it3.test.score325=
as.matrix(scale(cd.data[-cd.sample3,31:9246],center=T,scale=T))%*%
cd.cpca.f1.it3.loading325
cd.cpca.f1.it3.test.score34=
as.matrix(scale(cd.data[-cd.sample3,31:9246],center=T,scale=T))%*%
cd.cpca.f1.it3.loading34

#Support Vector Regression Models with overal RMSE Estimates
cd.cpca.f1.it3.325.rmse.vec=vector(length=30)

for (i in 1:30){
output<-cd.data[cd.sample3,i]
model.data<-as.data.frame(cbind(output,cd.cpca.f1.it3.score325))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,cd.cpca.f1.it3.test.score325)
cd.cpca.f1.it3.325.rmse.vec[i]<-
sqrt(mean((svr.pred-(cd.data[-cd.sample3,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(cd.cpca.f1.it3.325.rmse.vec,ncol=2,byrow=T))^2))

cd.cpca.f1.it3.34.rmse.vec=vector(length=30)
for (i in 1:30){
output<-cd.data[cd.sample3,i]
model.data<-as.data.frame(cbind(output,cd.cpca.f1.it3.score34))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,cd.cpca.f1.it3.test.score34)
cd.cpca.f1.it3.34.rmse.vec[i]<-
sqrt(mean((svr.pred-(cd.data[-cd.sample3,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(cd.cpca.f1.it3.34.rmse.vec,ncol=2,byrow=T))^2))



#Fold 2, Iteration 3
ptm <- proc.time()
cd.cpca.f2.it3=
prcomp(cd.data[-cd.sample3,31:9246],center=T,scale=T)
proc.time() - ptm
summary(cd.cpca.f2.it3)

screeplot(cd.cpca.f2.it3, type="l",npcs=50)
abline(h=0,col="blue")
length(which((cd.cpca.f2.it3$sdev)^2>=1))

# 323 PC's under eigenvalues
# 32 PC's under Scree-plot

#Extracting loadings and score matrices
cd.cpca.f2.it3.loading323=cd.cpca.f2.it3$rotation[,1:323]
cd.cpca.f2.it3.loading32=cd.cpca.f2.it3$rotation[,1:32]
cd.cpca.f2.it3.score323=cd.cpca.f2.it3$x[,1:323]
cd.cpca.f2.it3.score32=cd.cpca.f2.it3$x[,1:32]

#Transforming test data into pca-scores
cd.cpca.f2.it3.test.score323=
as.matrix(scale(cd.data[cd.sample3,31:9246],center=T,scale=T))%*%
cd.cpca.f2.it3.loading323
cd.cpca.f2.it3.test.score32=
as.matrix(scale(cd.data[cd.sample3,31:9246],center=T,scale=T))%*%
cd.cpca.f2.it3.loading32

#Support Vector Regression Models with overal RMSE Estimates

cd.cpca.f2.it3.323.rmse.vec=vector(length=30)
for (i in 1:30){
output<-cd.data[-cd.sample3,i]
model.data<-as.data.frame(cbind(output,cd.cpca.f2.it3.score323))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,cd.cpca.f2.it3.test.score323)
cd.cpca.f2.it3.323.rmse.vec[i]<-
sqrt(mean((svr.pred-(cd.data[cd.sample3,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(cd.cpca.f2.it3.323.rmse.vec,ncol=2,byrow=T))^2))

cd.cpca.f2.it3.32.rmse.vec=vector(length=30)
for (i in 1:30){
output<-cd.data[-cd.sample3,i]
model.data<-as.data.frame(cbind(output,cd.cpca.f2.it3.score32))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,cd.cpca.f2.it3.test.score32)
cd.cpca.f2.it3.32.rmse.vec[i]<-
sqrt(mean((svr.pred-(cd.data[cd.sample3,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(cd.cpca.f2.it3.32.rmse.vec,ncol=2,byrow=T))^2))


#Fold 1, Iteration 4
ptm <- proc.time()
cd.cpca.f1.it4=
prcomp(cd.data[cd.sample4,31:9246],center=T,scale=T)
proc.time() - ptm
summary(cd.cpca.f1.it4)

screeplot(cd.cpca.f1.it4, type="l",npcs=50)
abline(h=0,col="blue")
length(which((cd.cpca.f1.it4$sdev)^2>=1))

# 322 PC's under eigenvalues
# 33 PC's under Scree-plot

#Extracting loadings and score matrices
cd.cpca.f1.it4.loading322=cd.cpca.f1.it4$rotation[,1:322]
cd.cpca.f1.it4.loading33=cd.cpca.f1.it4$rotation[,1:33]
cd.cpca.f1.it4.score322=cd.cpca.f1.it4$x[,1:322]
cd.cpca.f1.it4.score33=cd.cpca.f1.it4$x[,1:33]

#Transforming test data into pca-scores
cd.cpca.f1.it4.test.score322=
as.matrix(scale(cd.data[-cd.sample4,31:9246],center=T,scale=T))%*%
cd.cpca.f1.it4.loading322
cd.cpca.f1.it4.test.score33=
as.matrix(scale(cd.data[-cd.sample4,31:9246],center=T,scale=T))%*%
cd.cpca.f1.it4.loading33

#Support Vector Regression Models with overal RMSE Estimates
cd.cpca.f1.it4.322.rmse.vec=vector(length=30)

for (i in 1:30){
output<-cd.data[cd.sample4,i]
model.data<-as.data.frame(cbind(output,cd.cpca.f1.it4.score322))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,cd.cpca.f1.it4.test.score322)
cd.cpca.f1.it4.322.rmse.vec[i]<-
sqrt(mean((svr.pred-(cd.data[-cd.sample4,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(cd.cpca.f1.it4.322.rmse.vec,ncol=2,byrow=T))^2))

cd.cpca.f1.it4.33.rmse.vec=vector(length=30)
for (i in 1:30){
output<-cd.data[cd.sample4,i]
model.data<-as.data.frame(cbind(output,cd.cpca.f1.it4.score33))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,cd.cpca.f1.it4.test.score33)
cd.cpca.f1.it4.33.rmse.vec[i]<-
sqrt(mean((svr.pred-(cd.data[-cd.sample4,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(cd.cpca.f1.it4.33.rmse.vec,ncol=2,byrow=T))^2))



#Fold 2, Iteration 4
ptm <- proc.time()
cd.cpca.f2.it4=
prcomp(cd.data[-cd.sample4,31:9246],center=T,scale=T)
proc.time() - ptm
summary(cd.cpca.f2.it4)

screeplot(cd.cpca.f2.it4, type="l",npcs=50)
abline(h=0,col="blue")
length(which((cd.cpca.f2.it4$sdev)^2>=1))

# 328 PC's under eigenvalues
# 39 PC's under Scree-plot

#Extracting loadings and score matrices
cd.cpca.f2.it4.loading328=cd.cpca.f2.it4$rotation[,1:328]
cd.cpca.f2.it4.loading39=cd.cpca.f2.it4$rotation[,1:39]
cd.cpca.f2.it4.score328=cd.cpca.f2.it4$x[,1:328]
cd.cpca.f2.it4.score39=cd.cpca.f2.it4$x[,1:39]

#Transforming test data into pca-scores
cd.cpca.f2.it4.test.score328=
as.matrix(scale(cd.data[cd.sample4,31:9246],center=T,scale=T))%*%
cd.cpca.f2.it4.loading328
cd.cpca.f2.it4.test.score39=
as.matrix(scale(cd.data[cd.sample4,31:9246],center=T,scale=T))%*%
cd.cpca.f2.it4.loading39

#Support Vector Regression Models with overal RMSE Estimates

cd.cpca.f2.it4.328.rmse.vec=vector(length=30)
for (i in 1:30){
output<-cd.data[-cd.sample4,i]
model.data<-as.data.frame(cbind(output,cd.cpca.f2.it4.score328))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,cd.cpca.f2.it4.test.score328)
cd.cpca.f2.it4.328.rmse.vec[i]<-
sqrt(mean((svr.pred-(cd.data[cd.sample4,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(cd.cpca.f2.it4.328.rmse.vec,ncol=2,byrow=T))^2))

cd.cpca.f2.it4.39.rmse.vec=vector(length=30)
for (i in 1:30){
output<-cd.data[-cd.sample4,i]
model.data<-as.data.frame(cbind(output,cd.cpca.f2.it4.score39))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,cd.cpca.f2.it4.test.score39)
cd.cpca.f2.it4.39.rmse.vec[i]<-
sqrt(mean((svr.pred-(cd.data[cd.sample4,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(cd.cpca.f2.it4.39.rmse.vec,ncol=2,byrow=T))^2))



#Fold 1, Iteration 5
ptm <- proc.time()
cd.cpca.f1.it5=
prcomp(cd.data[cd.sample5,31:9246],center=T,scale=T)
proc.time() - ptm
summary(cd.cpca.f1.it5)

screeplot(cd.cpca.f1.it5, type="l",npcs=50)
abline(h=0,col="blue")
length(which((cd.cpca.f1.it5$sdev)^2>=1))

# 325 PC's under eigenvalues
# 39 PC's under Scree-plot

#Extracting loadings and score matrices
cd.cpca.f1.it5.loading325=cd.cpca.f1.it5$rotation[,1:325]
cd.cpca.f1.it5.loading39=cd.cpca.f1.it5$rotation[,1:39]
cd.cpca.f1.it5.score325=cd.cpca.f1.it5$x[,1:325]
cd.cpca.f1.it5.score39=cd.cpca.f1.it5$x[,1:39]

#Transforming test data into pca-scores
cd.cpca.f1.it5.test.score325=
as.matrix(scale(cd.data[-cd.sample5,31:9246],center=T,scale=T))%*%
cd.cpca.f1.it5.loading325
cd.cpca.f1.it5.test.score39=
as.matrix(scale(cd.data[-cd.sample5,31:9246],center=T,scale=T))%*%
cd.cpca.f1.it5.loading39

#Support Vector Regression Models with overal RMSE Estimates
cd.cpca.f1.it5.325.rmse.vec=vector(length=30)

for (i in 1:30){
output<-cd.data[cd.sample5,i]
model.data<-as.data.frame(cbind(output,cd.cpca.f1.it5.score325))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,cd.cpca.f1.it5.test.score325)
cd.cpca.f1.it5.325.rmse.vec[i]<-
sqrt(mean((svr.pred-(cd.data[-cd.sample5,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(cd.cpca.f1.it5.325.rmse.vec,ncol=2,byrow=T))^2))

cd.cpca.f1.it5.39.rmse.vec=vector(length=30)
for (i in 1:30){
output<-cd.data[cd.sample5,i]
model.data<-as.data.frame(cbind(output,cd.cpca.f1.it5.score39))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,cd.cpca.f1.it5.test.score39)
cd.cpca.f1.it5.39.rmse.vec[i]<-
sqrt(mean((svr.pred-(cd.data[-cd.sample5,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(cd.cpca.f1.it5.39.rmse.vec,ncol=2,byrow=T))^2))



#Fold 2, Iteration 5
ptm <- proc.time()
cd.cpca.f2.it5=
prcomp(cd.data[-cd.sample5,31:9246],center=T,scale=T)
proc.time() - ptm
summary(cd.cpca.f2.it5)

screeplot(cd.cpca.f2.it5, type="l",npcs=50)
abline(h=0,col="blue")
length(which((cd.cpca.f2.it5$sdev)^2>=1))

# 326 PC's under eigenvalues
# 33 PC's under Scree-plot

#Extracting loadings and score matrices
cd.cpca.f2.it5.loading326=cd.cpca.f2.it5$rotation[,1:326]
cd.cpca.f2.it5.loading33=cd.cpca.f2.it5$rotation[,1:33]
cd.cpca.f2.it5.score326=cd.cpca.f2.it5$x[,1:326]
cd.cpca.f2.it5.score33=cd.cpca.f2.it5$x[,1:33]

#Transforming test data into pca-scores
cd.cpca.f2.it5.test.score326=
as.matrix(scale(cd.data[cd.sample5,31:9246],center=T,scale=T))%*%
cd.cpca.f2.it5.loading326
cd.cpca.f2.it5.test.score33=
as.matrix(scale(cd.data[cd.sample5,31:9246],center=T,scale=T))%*%
cd.cpca.f2.it5.loading33

#Support Vector Regression Models with overal RMSE Estimates

cd.cpca.f2.it5.326.rmse.vec=vector(length=30)
for (i in 1:30){
output<-cd.data[-cd.sample5,i]
model.data<-as.data.frame(cbind(output,cd.cpca.f2.it5.score326))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,cd.cpca.f2.it5.test.score326)
cd.cpca.f2.it5.326.rmse.vec[i]<-
sqrt(mean((svr.pred-(cd.data[cd.sample5,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(cd.cpca.f2.it5.326.rmse.vec,ncol=2,byrow=T))^2))

cd.cpca.f2.it5.33.rmse.vec=vector(length=30)
for (i in 1:30){
output<-cd.data[-cd.sample5,i]
model.data<-as.data.frame(cbind(output,cd.cpca.f2.it5.score33))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,cd.cpca.f2.it5.test.score33)
cd.cpca.f2.it5.33.rmse.vec[i]<-
sqrt(mean((svr.pred-(cd.data[cd.sample5,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(cd.cpca.f2.it5.33.rmse.vec,ncol=2,byrow=T))^2))



#Hubert's PCA with CD-dataset


#Fold 1, Iteration 1
ptm <- proc.time()
cd.hpca.f1.it1=
PcaHubert(cd.data[cd.sample1,31:9246],center=T,scale=T,kmax=9216)
proc.time() - ptm
summary(cd.hpca.f1.it1)

screeplot(cd.hpca.f1.it1, type="l",npcs=22,main="cd.hpca.f1.it1")
abline(h=0,col="blue")

# 22 PC's under all
# 11 PC's under Scree-plot

#Extracting loadings and score matrices
cd.hpca.f1.it1.loading22=cd.hpca.f1.it1@loadings[,1:22]
cd.hpca.f1.it1.loading11=cd.hpca.f1.it1@loadings[,1:11]
cd.hpca.f1.it1.score22=cd.hpca.f1.it1@scores[,1:22]
cd.hpca.f1.it1.score11=cd.hpca.f1.it1@scores[,1:11]

#Transforming test data into pca-scores
cd.hpca.f1.it1.test.score22=
as.matrix(scale(cd.data[-cd.sample1,31:9246],center=T,scale=T))%*%
cd.hpca.f1.it1.loading22
cd.hpca.f1.it1.test.score11=
as.matrix(scale(cd.data[-cd.sample1,31:9246],center=T,scale=T))%*%
cd.hpca.f1.it1.loading11

#Support Vector Regression Models with overal RMSE Estimates
cd.hpca.f1.it1.22.rmse.vec=vector(length=30)

for (i in 1:30){
output<-cd.data[cd.sample1,i]
model.data<-as.data.frame(cbind(output,cd.hpca.f1.it1.score22))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,cd.hpca.f1.it1.test.score22)
cd.hpca.f1.it1.22.rmse.vec[i]<-
sqrt(mean((svr.pred-(cd.data[-cd.sample1,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(cd.hpca.f1.it1.22.rmse.vec,ncol=2,byrow=T))^2))

cd.hpca.f1.it1.11.rmse.vec=vector(length=30)

for (i in 1:30){
output<-cd.data[cd.sample1,i]
model.data<-as.data.frame(cbind(output,cd.hpca.f1.it1.score11))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,cd.hpca.f1.it1.test.score11)
cd.hpca.f1.it1.11.rmse.vec[i]<-
sqrt(mean((svr.pred-(cd.data[-cd.sample1,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(cd.hpca.f1.it1.11.rmse.vec,ncol=2,byrow=T))^2))



#Fold 2 , Iteration 1
ptm <- proc.time()
cd.hpca.f2.it1=
PcaHubert(cd.data[-cd.sample1,31:9246],center=T,scale=T,kmax=9216)
proc.time() - ptm
summary(cd.hpca.f2.it1)

screeplot(cd.hpca.f2.it1, type="l",npcs=22,main="cd.hpca.f2.it1")
abline(h=0,col="blue")

# 22 PC's under all
# 11 PC's under Scree-plot

#Extracting loadings and score matrices
cd.hpca.f2.it1.loading22=cd.hpca.f2.it1@loadings[,1:22]
cd.hpca.f2.it1.loading11=cd.hpca.f2.it1@loadings[,1:11]
cd.hpca.f2.it1.score22=cd.hpca.f2.it1@scores[,1:22]
cd.hpca.f2.it1.score11=cd.hpca.f2.it1@scores[,1:11]

#Transforming test data into pca-scores
cd.hpca.f2.it1.test.score22=
as.matrix(scale(cd.data[cd.sample1,31:9246],center=T,scale=T))%*%
cd.hpca.f2.it1.loading22
cd.hpca.f2.it1.test.score11=
as.matrix(scale(cd.data[cd.sample1,31:9246],center=T,scale=T))%*%
cd.hpca.f2.it1.loading11

#Support Vector Regression Models with overal RMSE Estimates
cd.hpca.f2.it1.22.rmse.vec=vector(length=30)

for (i in 1:30){
output<-cd.data[-cd.sample1,i]
model.data<-as.data.frame(cbind(output,cd.hpca.f2.it1.score22))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,cd.hpca.f2.it1.test.score22)
cd.hpca.f2.it1.22.rmse.vec[i]<-
sqrt(mean((svr.pred-(cd.data[cd.sample1,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(cd.hpca.f2.it1.22.rmse.vec,ncol=2,byrow=T))^2))

cd.hpca.f2.it1.11.rmse.vec=vector(length=30)

for (i in 1:30){
output<-cd.data[-cd.sample1,i]
model.data<-as.data.frame(cbind(output,cd.hpca.f2.it1.score11))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,cd.hpca.f2.it1.test.score11)
cd.hpca.f2.it1.11.rmse.vec[i]<-
sqrt(mean((svr.pred-(cd.data[cd.sample1,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(cd.hpca.f2.it1.11.rmse.vec,ncol=2,byrow=T))^2))



#Fold 1, Iteration 2
ptm <- proc.time()
cd.hpca.f1.it2=
PcaHubert(cd.data[cd.sample2,31:9246],center=T,scale=T,kmax=9216)
proc.time() - ptm
summary(cd.hpca.f1.it2)

screeplot(cd.hpca.f1.it2, type="l",npcs=22,main="cd.hpca.f1.it2")
abline(h=0,col="blue")

# 22 PC's under all
# 13 PC's under Scree-plot

#Extracting loadings and score matrices
cd.hpca.f1.it2.loading22=cd.hpca.f1.it2@loadings[,1:22]
cd.hpca.f1.it2.loading13=cd.hpca.f1.it2@loadings[,1:13]
cd.hpca.f1.it2.score22=cd.hpca.f1.it2@scores[,1:22]
cd.hpca.f1.it2.score13=cd.hpca.f1.it2@scores[,1:13]

#Transforming test data into pca-scores
cd.hpca.f1.it2.test.score22=
as.matrix(scale(cd.data[-cd.sample2,31:9246],center=T,scale=T))%*%
cd.hpca.f1.it2.loading22
cd.hpca.f1.it2.test.score13=
as.matrix(scale(cd.data[-cd.sample2,31:9246],center=T,scale=T))%*%
cd.hpca.f1.it2.loading13

#Support Vector Regression Models with overal RMSE Estimates
cd.hpca.f1.it2.22.rmse.vec=vector(length=30)

for (i in 1:30){
output<-cd.data[cd.sample2,i]
model.data<-as.data.frame(cbind(output,cd.hpca.f1.it2.score22))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,cd.hpca.f1.it2.test.score22)
cd.hpca.f1.it2.22.rmse.vec[i]<-
sqrt(mean((svr.pred-(cd.data[-cd.sample2,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(cd.hpca.f1.it2.22.rmse.vec,ncol=2,byrow=T))^2))

cd.hpca.f1.it2.13.rmse.vec=vector(length=30)

for (i in 1:30){
output<-cd.data[cd.sample2,i]
model.data<-as.data.frame(cbind(output,cd.hpca.f1.it2.score13))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,cd.hpca.f1.it2.test.score13)
cd.hpca.f1.it2.13.rmse.vec[i]<-
sqrt(mean((svr.pred-(cd.data[-cd.sample2,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(cd.hpca.f1.it2.13.rmse.vec,ncol=2,byrow=T))^2))



#Fold 2, Iteration 2
ptm <- proc.time()
cd.hpca.f2.it2=
PcaHubert(cd.data[-cd.sample2,31:9246],center=T,scale=T,kmax=9216)
proc.time() - ptm
summary(cd.hpca.f2.it2)

screeplot(cd.hpca.f2.it2, type="l",npcs=22,main="cd.hpca.f2.it2")
abline(h=0,col="blue")

# 22 PC's under all
# 13 PC's under Scree-plot

#Extracting loadings and score matrices
cd.hpca.f2.it2.loading22=cd.hpca.f2.it2@loadings[,1:22]
cd.hpca.f2.it2.loading13=cd.hpca.f2.it2@loadings[,1:13]
cd.hpca.f2.it2.score22=cd.hpca.f2.it2@scores[,1:22]
cd.hpca.f2.it2.score13=cd.hpca.f2.it2@scores[,1:13]

#Transforming test data into pca-scores
cd.hpca.f2.it2.test.score22=
as.matrix(scale(cd.data[cd.sample2,31:9246],center=T,scale=T))%*%
cd.hpca.f2.it2.loading22
cd.hpca.f2.it2.test.score13=
as.matrix(scale(cd.data[cd.sample2,31:9246],center=T,scale=T))%*%
cd.hpca.f2.it2.loading13

#Support Vector Regression Models with overal RMSE Estimates
cd.hpca.f2.it2.22.rmse.vec=vector(length=30)

for (i in 1:30){
output<-cd.data[-cd.sample2,i]
model.data<-as.data.frame(cbind(output,cd.hpca.f2.it2.score22))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,cd.hpca.f2.it2.test.score22)
cd.hpca.f2.it2.22.rmse.vec[i]<-
sqrt(mean((svr.pred-(cd.data[cd.sample2,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(cd.hpca.f2.it2.22.rmse.vec,ncol=2,byrow=T))^2))

cd.hpca.f2.it2.13.rmse.vec=vector(length=30)

for (i in 1:30){
output<-cd.data[-cd.sample2,i]
model.data<-as.data.frame(cbind(output,cd.hpca.f2.it2.score13))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,cd.hpca.f2.it2.test.score13)
cd.hpca.f2.it2.13.rmse.vec[i]<-
sqrt(mean((svr.pred-(cd.data[cd.sample2,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(cd.hpca.f2.it2.13.rmse.vec,ncol=2,byrow=T))^2))



#Fold 1, Iteration 3
ptm <- proc.time()
cd.hpca.f1.it3=
PcaHubert(cd.data[cd.sample3,31:9246],center=T,scale=T,kmax=9216)
proc.time() - ptm
summary(cd.hpca.f1.it3)

screeplot(cd.hpca.f1.it3, type="l",npcs=23,main="cd.hpca.f1.it3")
abline(h=0,col="blue")

# 23 PC's under all
# 12 PC's under Scree-plot

#Extracting loadings and score matrices
cd.hpca.f1.it3.loading23=cd.hpca.f1.it3@loadings[,1:23]
cd.hpca.f1.it3.loading12=cd.hpca.f1.it3@loadings[,1:12]
cd.hpca.f1.it3.score23=cd.hpca.f1.it3@scores[,1:23]
cd.hpca.f1.it3.score12=cd.hpca.f1.it3@scores[,1:12]

#Transforming test data into pca-scores
cd.hpca.f1.it3.test.score23=
as.matrix(scale(cd.data[-cd.sample3,31:9246],center=T,scale=T))%*%
cd.hpca.f1.it3.loading23
cd.hpca.f1.it3.test.score12=
as.matrix(scale(cd.data[-cd.sample3,31:9246],center=T,scale=T))%*%
cd.hpca.f1.it3.loading12

#Support Vector Regression Models with overal RMSE Estimates
cd.hpca.f1.it3.23.rmse.vec=vector(length=30)

for (i in 1:30){
output<-cd.data[cd.sample3,i]
model.data<-as.data.frame(cbind(output,cd.hpca.f1.it3.score23))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,cd.hpca.f1.it3.test.score23)
cd.hpca.f1.it3.23.rmse.vec[i]<-
sqrt(mean((svr.pred-(cd.data[-cd.sample3,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(cd.hpca.f1.it3.23.rmse.vec,ncol=2,byrow=T))^2))

cd.hpca.f1.it3.12.rmse.vec=vector(length=30)

for (i in 1:30){
output<-cd.data[cd.sample3,i]
model.data<-as.data.frame(cbind(output,cd.hpca.f1.it3.score12))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,cd.hpca.f1.it3.test.score12)
cd.hpca.f1.it3.12.rmse.vec[i]<-
sqrt(mean((svr.pred-(cd.data[-cd.sample3,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(cd.hpca.f1.it3.12.rmse.vec,ncol=2,byrow=T))^2))



#Fold 2, Iteration 3
ptm <- proc.time()
cd.hpca.f2.it3=
PcaHubert(cd.data[-cd.sample3,31:9246],center=T,scale=T,kmax=9216)
proc.time() - ptm
summary(cd.hpca.f2.it3)

screeplot(cd.hpca.f2.it3, type="l",npcs=21,main="cd.hpca.f2.it3")
abline(h=0,col="blue")

# 21 PC's under all
# 13 PC's under Scree-plot

#Extracting loadings and score matrices
cd.hpca.f2.it3.loading21=cd.hpca.f2.it3@loadings[,1:21]
cd.hpca.f2.it3.loading13=cd.hpca.f2.it3@loadings[,1:13]
cd.hpca.f2.it3.score21=cd.hpca.f2.it3@scores[,1:21]
cd.hpca.f2.it3.score13=cd.hpca.f2.it3@scores[,1:13]

#Transforming test data into pca-scores
cd.hpca.f2.it3.test.score21=
as.matrix(scale(cd.data[cd.sample3,31:9246],center=T,scale=T))%*%
cd.hpca.f2.it3.loading21
cd.hpca.f2.it3.test.score13=
as.matrix(scale(cd.data[cd.sample3,31:9246],center=T,scale=T))%*%
cd.hpca.f2.it3.loading13

#Support Vector Regression Models with overal RMSE Estimates

cd.hpca.f2.it3.21.rmse.vec=vector(length=30)
for (i in 1:30){
output<-cd.data[-cd.sample3,i]
model.data<-as.data.frame(cbind(output,cd.hpca.f2.it3.score21))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,cd.hpca.f2.it3.test.score21)
cd.hpca.f2.it3.21.rmse.vec[i]<-
sqrt(mean((svr.pred-(cd.data[cd.sample3,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(cd.hpca.f2.it3.21.rmse.vec,ncol=2,byrow=T))^2))


cd.hpca.f2.it3.13.rmse.vec=vector(length=30)
for (i in 1:30){
output<-cd.data[-cd.sample3,i]
model.data<-as.data.frame(cbind(output,cd.hpca.f2.it3.score13))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,cd.hpca.f2.it3.test.score13)
cd.hpca.f2.it3.13.rmse.vec[i]<-
sqrt(mean((svr.pred-(cd.data[cd.sample3,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(cd.hpca.f2.it3.13.rmse.vec,ncol=2,byrow=T))^2))


#Fold 1, Iteration 4
ptm <- proc.time()
cd.hpca.f1.it4=PcaHubert(cd.data[cd.sample4,31:9246],center=T,scale=T,kmax=9216)
proc.time() - ptm
summary(cd.hpca.f1.it4)

screeplot(cd.hpca.f1.it4, type="l",npcs=22,main="cd.hpca.f1.it4")
abline(h=0,col="blue")

# 22 PC's under all
# 13 PC's under Scree-plot

#Extracting loadings and score matrices
cd.hpca.f1.it4.loading22=cd.hpca.f1.it4@loadings[,1:22]
cd.hpca.f1.it4.loading13=cd.hpca.f1.it4@loadings[,1:13]
cd.hpca.f1.it4.score22=cd.hpca.f1.it4@scores[,1:22]
cd.hpca.f1.it4.score13=cd.hpca.f1.it4@scores[,1:13]

#Transforming test data into pca-scores
cd.hpca.f1.it4.test.score22=
as.matrix(scale(cd.data[-cd.sample4,31:9246],center=T,scale=T))%*%
cd.hpca.f1.it4.loading22
cd.hpca.f1.it4.test.score13=
as.matrix(scale(cd.data[-cd.sample4,31:9246],center=T,scale=T))%*%
cd.hpca.f1.it4.loading13

#Support Vector Regression Models with overal RMSE Estimates
cd.hpca.f1.it4.22.rmse.vec=vector(length=30)

for (i in 1:30){
output<-cd.data[cd.sample4,i]
model.data<-as.data.frame(cbind(output,cd.hpca.f1.it4.score22))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,cd.hpca.f1.it4.test.score22)
cd.hpca.f1.it4.22.rmse.vec[i]<-
sqrt(mean((svr.pred-(cd.data[-cd.sample4,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(cd.hpca.f1.it4.22.rmse.vec,ncol=2,byrow=T))^2))


cd.hpca.f1.it4.13.rmse.vec=vector(length=30)

for (i in 1:30){
output<-cd.data[cd.sample4,i]
model.data<-as.data.frame(cbind(output,cd.hpca.f1.it4.score13))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,cd.hpca.f1.it4.test.score13)
cd.hpca.f1.it4.13.rmse.vec[i]<-
sqrt(mean((svr.pred-(cd.data[-cd.sample4,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(cd.hpca.f1.it4.13.rmse.vec,ncol=2,byrow=T))^2))



#Fold 2, Iteration 4
ptm <- proc.time()
cd.hpca.f2.it4=
PcaHubert(cd.data[-cd.sample4,31:9246],center=T,scale=T,kmax=9216)
proc.time() - ptm
summary(cd.hpca.f2.it4)

screeplot(cd.hpca.f2.it4, type="l",npcs=23,main="cd.hpca.f2.it4")
abline(h=0,col="blue")

# 23 PC's under all
# 13 PC's under Scree-plot

#Extracting loadings and score matrices
cd.hpca.f2.it4.loading23=cd.hpca.f2.it4@loadings[,1:23]
cd.hpca.f2.it4.loading13=cd.hpca.f2.it4@loadings[,1:13]
cd.hpca.f2.it4.score23=cd.hpca.f2.it4@scores[,1:23]
cd.hpca.f2.it4.score13=cd.hpca.f2.it4@scores[,1:13]

#Transforming test data into pca-scores
cd.hpca.f2.it4.test.score23=
as.matrix(scale(cd.data[cd.sample4,31:9246],center=T,scale=T))%*%
cd.hpca.f2.it4.loading23
cd.hpca.f2.it4.test.score13=
as.matrix(scale(cd.data[cd.sample4,31:9246],center=T,scale=T))%*%
cd.hpca.f2.it4.loading13

#Support Vector Regression Models with overal RMSE Estimates

cd.hpca.f2.it4.23.rmse.vec=vector(length=30)
for (i in 1:30){
output<-cd.data[-cd.sample4,i]
model.data<-as.data.frame(cbind(output,cd.hpca.f2.it4.score23))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,cd.hpca.f2.it4.test.score23)
cd.hpca.f2.it4.23.rmse.vec[i]<-
sqrt(mean((svr.pred-(cd.data[cd.sample4,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(cd.hpca.f2.it4.23.rmse.vec,ncol=2,byrow=T))^2))


cd.hpca.f2.it4.13.rmse.vec=vector(length=30)
for (i in 1:30){
output<-cd.data[-cd.sample4,i]
model.data<-as.data.frame(cbind(output,cd.hpca.f2.it4.score13))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,cd.hpca.f2.it4.test.score13)
cd.hpca.f2.it4.13.rmse.vec[i]<-
sqrt(mean((svr.pred-(cd.data[cd.sample4,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(cd.hpca.f2.it4.13.rmse.vec,ncol=2,byrow=T))^2))



#Fold 1, Iteration 5
ptm <- proc.time()
cd.hpca.f1.it5=
PcaHubert(cd.data[cd.sample5,31:9246],center=T,scale=T,kmax=9216)
proc.time() - ptm
summary(cd.hpca.f1.it5)

screeplot(cd.hpca.f1.it5, type="l",npcs=22,main="cd.hpca.f1.it5")
abline(h=0,col="blue")

# 22 PC's under all
# 13 PC's under Scree-plot

#Extracting loadings and score matrices
cd.hpca.f1.it5.loading22=cd.hpca.f1.it5@loadings[,1:22]
cd.hpca.f1.it5.loading13=cd.hpca.f1.it5@loadings[,1:13]
cd.hpca.f1.it5.score22=cd.hpca.f1.it5@scores[,1:22]
cd.hpca.f1.it5.score13=cd.hpca.f1.it5@scores[,1:13]

#Transforming test data into pca-scores
cd.hpca.f1.it5.test.score22=
as.matrix(scale(cd.data[-cd.sample5,31:9246],center=T,scale=T))%*%
cd.hpca.f1.it5.loading22
cd.hpca.f1.it5.test.score13=
as.matrix(scale(cd.data[-cd.sample5,31:9246],center=T,scale=T))%*%
cd.hpca.f1.it5.loading13

#Support Vector Regression Models with overal RMSE Estimates
cd.hpca.f1.it5.22.rmse.vec=vector(length=30)

for (i in 1:30){
output<-cd.data[cd.sample5,i]
model.data<-as.data.frame(cbind(output,cd.hpca.f1.it5.score22))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,cd.hpca.f1.it5.test.score22)
cd.hpca.f1.it5.22.rmse.vec[i]<-
sqrt(mean((svr.pred-(cd.data[-cd.sample5,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(cd.hpca.f1.it5.22.rmse.vec,ncol=2,byrow=T))^2))


cd.hpca.f1.it5.13.rmse.vec=vector(length=30)

for (i in 1:30){
output<-cd.data[cd.sample5,i]
model.data<-as.data.frame(cbind(output,cd.hpca.f1.it5.score13))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,cd.hpca.f1.it5.test.score13)
cd.hpca.f1.it5.13.rmse.vec[i]<-
sqrt(mean((svr.pred-(cd.data[-cd.sample5,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(cd.hpca.f1.it5.13.rmse.vec,ncol=2,byrow=T))^2))



#Fold 2, Iteration 5
ptm <- proc.time()
cd.hpca.f2.it5=
PcaHubert(cd.data[-cd.sample5,31:9246],center=T,scale=T,kmax=9216)
proc.time() - ptm
summary(cd.hpca.f2.it5)

screeplot(cd.hpca.f2.it5, type="l",npcs=23,main="cd.hpca.f2.it5")
abline(h=0,col="blue")

# 23 PC's under all
# 13 PC's under Scree-plot

#Extracting loadings and score matrices
cd.hpca.f2.it5.loading23=cd.hpca.f2.it5@loadings[,1:23]
cd.hpca.f2.it5.loading13=cd.hpca.f2.it5@loadings[,1:13]
cd.hpca.f2.it5.score23=cd.hpca.f2.it5@scores[,1:23]
cd.hpca.f2.it5.score13=cd.hpca.f2.it5@scores[,1:13]

#Transforming test data into pca-scores
cd.hpca.f2.it5.test.score23=
as.matrix(scale(cd.data[cd.sample5,31:9246],center=T,scale=T))%*%
cd.hpca.f2.it5.loading23
cd.hpca.f2.it5.test.score13=
as.matrix(scale(cd.data[cd.sample5,31:9246],center=T,scale=T))%*%
cd.hpca.f2.it5.loading13

#Support Vector Regression Models with overal RMSE Estimates
cd.hpca.f2.it5.23.rmse.vec=vector(length=30)

for (i in 1:30){
output<-cd.data[-cd.sample5,i]
model.data<-as.data.frame(cbind(output,cd.hpca.f2.it5.score23))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,cd.hpca.f2.it5.test.score23)
cd.hpca.f2.it5.23.rmse.vec[i]<-
sqrt(mean((svr.pred-(cd.data[cd.sample5,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(cd.hpca.f2.it5.23.rmse.vec,ncol=2,byrow=T))^2))


cd.hpca.f2.it5.13.rmse.vec=vector(length=30)

for (i in 1:30){
output<-cd.data[-cd.sample5,i]
model.data<-as.data.frame(cbind(output,cd.hpca.f2.it5.score13))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,cd.hpca.f2.it5.test.score13)
cd.hpca.f2.it5.13.rmse.vec[i]<-
sqrt(mean((svr.pred-(cd.data[cd.sample5,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(cd.hpca.f2.it5.13.rmse.vec,ncol=2,byrow=T))^2))



#Mean Imputed Dataset
mu.data=meanimp(data)

#Mean Imputation Dataset Split 
#The indices generated here are also used for the ROBPCA 

mu.sample1 <- 
sample.int(n = 7049, size = floor(.50*7049), replace = F)
mu.sample2 <-
sample.int(n = 7049, size = floor(.50*7049), replace = F)
mu.sample3 <- 
sample.int(n = 7049, size = floor(.50*7049), replace = F)
mu.sample4 <- 
sample.int(n = 7049, size = floor(.50*7049), replace = F)
mu.sample5 <- 
sample.int(n = 7049, size = floor(.50*7049), replace = F)

#One fold will have 3524 obs and the other 3525 obs

mu.sample1 <-mu.sample1
mu.sample2 <-mu.sample2
mu.sample3 <-mu.sample3
mu.sample4 <-mu.sample4
mu.sample5 <-mu.sample5

 

###First fold, first iteration
#Perform CPCA and assess components by Scree-plot and Eigenvalues >=1
ptm <- proc.time()
mu.cpca.f1.it1=prcomp(image.data[mu.sample1,],center=T,scale=T)
proc.time() - ptm
summary(mu.cpca.f1.it1)

screeplot(mu.cpca.f1.it1, type="l",npcs=50)
abline(h=0,col="blue")
length(which((mu.cpca.f1.it1$sdev)^2>=1))

# 328 PC's under eigenvalues
# 30 PC's under Scree-plot

#Extracting loadings and score matrices
mu.cpca.f1.it1.loading328=mu.cpca.f1.it1$rotation[,1:328]
mu.cpca.f1.it1.loading30=mu.cpca.f1.it1$rotation[,1:30]
mu.cpca.f1.it1.score328=mu.cpca.f1.it1$x[,1:328]
mu.cpca.f1.it1.score30=mu.cpca.f1.it1$x[,1:30]

#Transforming test data into pca-scores
mu.cpca.f1.it1.test.score328=
as.matrix(scale(image.data[-mu.sample1,],center=T,scale=T))%*%
mu.cpca.f1.it1.loading328
mu.cpca.f1.it1.test.score30=
as.matrix(scale(image.data[-mu.sample1,],center=T,scale=T))%*%
mu.cpca.f1.it1.loading30

#Support Vector Regression Models with overal RMSE Estimates
mu.cpca.f1.it1.328.rmse.vec=vector(length=30)

for (i in 1:30){
output<-mu.data[mu.sample1,i]
model.data<-as.data.frame(cbind(output,mu.cpca.f1.it1.score328))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,mu.cpca.f1.it1.test.score328)
mu.cpca.f1.it1.328.rmse.vec[i]<-
sqrt(mean((svr.pred-(mu.data[-mu.sample1,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(mu.cpca.f1.it1.328.rmse.vec,ncol=2,byrow=T))^2))

mu.cpca.f1.it1.30.rmse.vec=vector(length=30)
for (i in 1:30){
output<-mu.data[mu.sample1,i]
model.data<-as.data.frame(cbind(output,mu.cpca.f1.it1.score30))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,mu.cpca.f1.it1.test.score30)
mu.cpca.f1.it1.30.rmse.vec[i]<-
sqrt(mean((svr.pred-(mu.data[-mu.sample1,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(mu.cpca.f1.it1.30.rmse.vec,ncol=2,byrow=T))^2))



###Second fold, first iteration
ptm <- proc.time()
mu.cpca.f2.it1=prcomp(image.data[-mu.sample1,],center=T,scale=T)
proc.time() - ptm
summary(mu.cpca.f2.it1)

screeplot(mu.cpca.f2.it1, type="l",npcs=50)
abline(h=0,col="blue")
length(which((mu.cpca.f2.it1$sdev)^2>=1))

# 325 PC's under eigenvalues
# 29 PC's under Scree-plot

#Extracting loadings and score matrices
mu.cpca.f2.it1.loading325=mu.cpca.f2.it1$rotation[,1:325]
mu.cpca.f2.it1.loading29=mu.cpca.f2.it1$rotation[,1:29]
mu.cpca.f2.it1.score325=mu.cpca.f2.it1$x[,1:325]
mu.cpca.f2.it1.score29=mu.cpca.f2.it1$x[,1:29]

#Transforming test data into pca-scores
mu.cpca.f2.it1.test.score325=
as.matrix(scale(image.data[mu.sample1,],center=T,scale=T))%*%
mu.cpca.f2.it1.loading325
mu.cpca.f2.it1.test.score29=
as.matrix(scale(image.data[mu.sample1,],center=T,scale=T))%*%
mu.cpca.f2.it1.loading29

#Support Vector Regression Models with overal RMSE Estimates
mu.cpca.f2.it1.325.rmse.vec=vector(length=30)

for (i in 1:30){
output<-mu.data[-mu.sample1,i]
model.data<-as.data.frame(cbind(output,mu.cpca.f2.it1.score325))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,mu.cpca.f2.it1.test.score325)
mu.cpca.f2.it1.325.rmse.vec[i]<-
sqrt(mean((svr.pred-(mu.data[mu.sample1,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(mu.cpca.f2.it1.325.rmse.vec,ncol=2,byrow=T))^2))

mu.cpca.f2.it1.29.rmse.vec=vector(length=30)
for (i in 1:30){
output<-mu.data[-mu.sample1,i]
model.data<-as.data.frame(cbind(output,mu.cpca.f2.it1.score29))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,mu.cpca.f2.it1.test.score29)
mu.cpca.f2.it1.29.rmse.vec[i]<-
sqrt(mean((svr.pred-(mu.data[mu.sample1,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(mu.cpca.f2.it1.29.rmse.vec,ncol=2,byrow=T))^2))



##First Fold, Second Iteration
ptm <- proc.time()
mu.cpca.f1.it2=prcomp(image.data[mu.sample2,],center=T,scale=T)
proc.time() - ptm
summary(mu.cpca.f1.it2)

screeplot(mu.cpca.f1.it2, type="l",npcs=50)
abline(h=0,col="blue")
length(which((mu.cpca.f1.it2$sdev)^2>=1))

# 328 PC's under eigenvalues
# 33 PC's under Scree-plot

#Extracting loadings and score matrices
mu.cpca.f1.it2.loading328=mu.cpca.f1.it2$rotation[,1:328]
mu.cpca.f1.it2.loading33=mu.cpca.f1.it2$rotation[,1:33]
mu.cpca.f1.it2.score328=mu.cpca.f1.it2$x[,1:328]
mu.cpca.f1.it2.score33=mu.cpca.f1.it2$x[,1:33]

#Transforming test data into pca-scores
mu.cpca.f1.it2.test.score328=
as.matrix(scale(image.data[-mu.sample2,],center=T,scale=T))%*%
mu.cpca.f1.it2.loading328
mu.cpca.f1.it2.test.score33=
as.matrix(scale(image.data[-mu.sample2,],center=T,scale=T))%*%
mu.cpca.f1.it2.loading33

#Support Vector Regression Models with overal RMSE Estimates
mu.cpca.f1.it2.328.rmse.vec=vector(length=30)

for (i in 1:30){
output<-mu.data[mu.sample2,i]
model.data<-as.data.frame(cbind(output,mu.cpca.f1.it2.score328))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,mu.cpca.f1.it2.test.score328)
mu.cpca.f1.it2.328.rmse.vec[i]<-
sqrt(mean((svr.pred-(mu.data[-mu.sample2,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(mu.cpca.f1.it2.328.rmse.vec,ncol=2,byrow=T))^2))

mu.cpca.f1.it2.33.rmse.vec=vector(length=30)
for (i in 1:30){
output<-mu.data[mu.sample2,i]
model.data<-as.data.frame(cbind(output,mu.cpca.f1.it2.score33))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,mu.cpca.f1.it2.test.score33)
mu.cpca.f1.it2.33.rmse.vec[i]<-
sqrt(mean((svr.pred-(mu.data[-mu.sample2,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(mu.cpca.f1.it2.33.rmse.vec,ncol=2,byrow=T))^2))



##Second Fold, Second Iteration
ptm <- proc.time()
mu.cpca.f2.it2=prcomp(image.data[-mu.sample2,],center=T,scale=T)
proc.time() - ptm
summary(mu.cpca.f2.it2)


screeplot(mu.cpca.f2.it2, type="l",npcs=50)
abline(h=0,col="blue")
length(which((mu.cpca.f2.it2$sdev)^2>=1))

# 323 PC's under eigenvalues
# 34 PC's under Scree-plot

#Extracting loadings and score matrices
mu.cpca.f2.it2.loading323=mu.cpca.f2.it2$rotation[,1:323]
mu.cpca.f2.it2.loading34=mu.cpca.f2.it2$rotation[,1:34]
mu.cpca.f2.it2.score323=mu.cpca.f2.it2$x[,1:323]
mu.cpca.f2.it2.score34=mu.cpca.f2.it2$x[,1:34]

#Transforming test data into pca-scores
mu.cpca.f2.it2.test.score323=
as.matrix(scale(image.data[mu.sample2,],center=T,scale=T))%*%
mu.cpca.f2.it2.loading323
mu.cpca.f2.it2.test.score34=
as.matrix(scale(image.data[mu.sample2,],center=T,scale=T))%*%
mu.cpca.f2.it2.loading34

#Support Vector Regression Models with overal RMSE Estimates
mu.cpca.f2.it2.323.rmse.vec=vector(length=30)

for (i in 1:30){
output<-mu.data[-mu.sample2,i]
model.data<-as.data.frame(cbind(output,mu.cpca.f2.it2.score323))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,mu.cpca.f2.it2.test.score323)
mu.cpca.f2.it2.323.rmse.vec[i]<-
sqrt(mean((svr.pred-(mu.data[mu.sample2,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(mu.cpca.f2.it2.323.rmse.vec,ncol=2,byrow=T))^2))

mu.cpca.f2.it2.34.rmse.vec=vector(length=30)
for (i in 1:30){
output<-mu.data[-mu.sample2,i]
model.data<-as.data.frame(cbind(output,mu.cpca.f2.it2.score34))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,mu.cpca.f2.it2.test.score34)
mu.cpca.f2.it2.34.rmse.vec[i]<-
sqrt(mean((svr.pred-(mu.data[mu.sample2,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(mu.cpca.f2.it2.34.rmse.vec,ncol=2,byrow=T))^2))



#Fold One, Iteration 3
ptm <- proc.time()
mu.cpca.f1.it3=prcomp(image.data[mu.sample3,],center=T,scale=T)
proc.time() - ptm
summary(mu.cpca.f1.it3)

screeplot(mu.cpca.f1.it3, type="l",npcs=50)
abline(h=0,col="blue")
length(which((mu.cpca.f1.it3$sdev)^2>=1))

# 324 PC's under eigenvalues
# 30 PC's under Scree-plot

#Extracting loadings and score matrices
mu.cpca.f1.it3.loading324=mu.cpca.f1.it3$rotation[,1:324]
mu.cpca.f1.it3.loading30=mu.cpca.f1.it3$rotation[,1:30]
mu.cpca.f1.it3.score324=mu.cpca.f1.it3$x[,1:324]
mu.cpca.f1.it3.score30=mu.cpca.f1.it3$x[,1:30]

#Transforming test data into pca-scores
mu.cpca.f1.it3.test.score324=
as.matrix(scale(image.data[-mu.sample3,],center=T,scale=T))%*%
mu.cpca.f1.it3.loading324
mu.cpca.f1.it3.test.score30=
as.matrix(scale(image.data[-mu.sample3,],center=T,scale=T))%*%
mu.cpca.f1.it3.loading30

#Support Vector Regression Models with overal RMSE Estimates
mu.cpca.f1.it3.324.rmse.vec=vector(length=30)

for (i in 1:30){
output<-mu.data[mu.sample3,i]
model.data<-as.data.frame(cbind(output,mu.cpca.f1.it3.score324))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,mu.cpca.f1.it3.test.score324)
mu.cpca.f1.it3.324.rmse.vec[i]<-
sqrt(mean((svr.pred-(mu.data[-mu.sample3,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(mu.cpca.f1.it3.324.rmse.vec,ncol=2,byrow=T))^2))

mu.cpca.f1.it3.30.rmse.vec=vector(length=30)
for (i in 1:30){
output<-mu.data[mu.sample3,i]
model.data<-as.data.frame(cbind(output,mu.cpca.f1.it3.score30))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,mu.cpca.f1.it3.test.score30)
mu.cpca.f1.it3.30.rmse.vec[i]<-
sqrt(mean((svr.pred-(mu.data[-mu.sample3,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(mu.cpca.f1.it3.30.rmse.vec,ncol=2,byrow=T))^2))



#Fold 2, Iteration 3
ptm <- proc.time()
mu.cpca.f2.it3=prcomp(image.data[-mu.sample3,],center=T,scale=T)
proc.time() - ptm
summary(mu.cpca.f2.it3)

screeplot(mu.cpca.f2.it3, type="l",npcs=50)
abline(h=0,col="blue")
length(which((mu.cpca.f2.it3$sdev)^2>=1))

# 329 PC's under eigenvalues
# 30 PC's under Scree-plot

#Extracting loadings and score matrices
mu.cpca.f2.it3.loading329=mu.cpca.f2.it3$rotation[,1:329]
mu.cpca.f2.it3.loading30=mu.cpca.f2.it3$rotation[,1:30]
mu.cpca.f2.it3.score329=mu.cpca.f2.it3$x[,1:329]
mu.cpca.f2.it3.score30=mu.cpca.f2.it3$x[,1:30]

#Transforming test data into pca-scores
mu.cpca.f2.it3.test.score329=
as.matrix(scale(image.data[mu.sample3,],center=T,scale=T))%*%
mu.cpca.f2.it3.loading329
mu.cpca.f2.it3.test.score30=
as.matrix(scale(image.data[mu.sample3,],center=T,scale=T))%*%
mu.cpca.f2.it3.loading30

#Support Vector Regression Models with overal RMSE Estimates

mu.cpca.f2.it3.329.rmse.vec=vector(length=30)
for (i in 1:30){
output<-mu.data[-mu.sample3,i]
model.data<-as.data.frame(cbind(output,mu.cpca.f2.it3.score329))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,mu.cpca.f2.it3.test.score329)
mu.cpca.f2.it3.329.rmse.vec[i]<-
sqrt(mean((svr.pred-(mu.data[mu.sample3,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(mu.cpca.f2.it3.329.rmse.vec,ncol=2,byrow=T))^2))

mu.cpca.f2.it3.30.rmse.vec=vector(length=30)
for (i in 1:30){
output<-mu.data[-mu.sample3,i]
model.data<-as.data.frame(cbind(output,mu.cpca.f2.it3.score30))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,mu.cpca.f2.it3.test.score30)
mu.cpca.f2.it3.30.rmse.vec[i]<-
sqrt(mean((svr.pred-(mu.data[mu.sample3,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(mu.cpca.f2.it3.30.rmse.vec,ncol=2,byrow=T))^2))



#Fold 1, Iteration 4
ptm <- proc.time()
mu.cpca.f1.it4=prcomp(image.data[mu.sample4,],center=T,scale=T)
proc.time() - ptm
summary(mu.cpca.f1.it4)

screeplot(mu.cpca.f1.it4, type="l",npcs=50)
abline(h=0,col="blue")
length(which((mu.cpca.f1.it4$sdev)^2>=1))

# 324 PC's under eigenvalues
# 33 PC's under Scree-plot

#Extracting loadings and score matrices
mu.cpca.f1.it4.loading324=mu.cpca.f1.it4$rotation[,1:324]
mu.cpca.f1.it4.loading33=mu.cpca.f1.it4$rotation[,1:33]
mu.cpca.f1.it4.score324=mu.cpca.f1.it4$x[,1:324]
mu.cpca.f1.it4.score33=mu.cpca.f1.it4$x[,1:33]

#Transforming test data into pca-scores
mu.cpca.f1.it4.test.score324=
as.matrix(scale(image.data[-mu.sample4,],center=T,scale=T))%*%
mu.cpca.f1.it4.loading324
mu.cpca.f1.it4.test.score33=
as.matrix(scale(image.data[-mu.sample4,],center=T,scale=T))%*%
mu.cpca.f1.it4.loading33

#Support Vector Regression Models with overal RMSE Estimates
mu.cpca.f1.it4.324.rmse.vec=vector(length=30)

for (i in 1:30){
output<-mu.data[mu.sample4,i]
model.data<-as.data.frame(cbind(output,mu.cpca.f1.it4.score324))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,mu.cpca.f1.it4.test.score324)
mu.cpca.f1.it4.324.rmse.vec[i]<-
sqrt(mean((svr.pred-(mu.data[-mu.sample4,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(mu.cpca.f1.it4.324.rmse.vec,ncol=2,byrow=T))^2))

mu.cpca.f1.it4.33.rmse.vec=vector(length=30)
for (i in 1:30){
output<-mu.data[mu.sample4,i]
model.data<-as.data.frame(cbind(output,mu.cpca.f1.it4.score33))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,mu.cpca.f1.it4.test.score33)
mu.cpca.f1.it4.33.rmse.vec[i]<-
sqrt(mean((svr.pred-(mu.data[-mu.sample4,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(mu.cpca.f1.it4.33.rmse.vec,ncol=2,byrow=T))^2))



#Fold 2, Iteration 4
ptm <- proc.time()
mu.cpca.f2.it4=prcomp(image.data[-mu.sample4,],center=T,scale=T)
proc.time() - ptm
summary(mu.cpca.f2.it4)

screeplot(mu.cpca.f2.it4, type="l",npcs=50)
abline(h=0,col="blue")
length(which((mu.cpca.f2.it4$sdev)^2>=1))

# 328 PC's under eigenvalues
# 30 PC's under Scree-plot

#Extracting loadings and score matrices
mu.cpca.f2.it4.loading328=mu.cpca.f2.it4$rotation[,1:328]
mu.cpca.f2.it4.loading30=mu.cpca.f2.it4$rotation[,1:30]
mu.cpca.f2.it4.score328=mu.cpca.f2.it4$x[,1:328]
mu.cpca.f2.it4.score30=mu.cpca.f2.it4$x[,1:30]

#Transforming test data into pca-scores
mu.cpca.f2.it4.test.score328=
as.matrix(scale(image.data[mu.sample4,],center=T,scale=T))%*%
mu.cpca.f2.it4.loading328
mu.cpca.f2.it4.test.score30=
as.matrix(scale(image.data[mu.sample4,],center=T,scale=T))%*%
mu.cpca.f2.it4.loading30

#Support Vector Regression Models with overal RMSE Estimates

mu.cpca.f2.it4.328.rmse.vec=vector(length=30)
for (i in 1:30){
output<-mu.data[-mu.sample4,i]
model.data<-as.data.frame(cbind(output,mu.cpca.f2.it4.score328))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,mu.cpca.f2.it4.test.score328)
mu.cpca.f2.it4.328.rmse.vec[i]<-
sqrt(mean((svr.pred-(mu.data[mu.sample4,i]))^2))
}
sqrt(mean(rowMeans(matrix(mu.cpca.f2.it4.328.rmse.vec,ncol=2,byrow=T))^2))

mu.cpca.f2.it4.30.rmse.vec=vector(length=30)
for (i in 1:30){
output<-mu.data[-mu.sample4,i]
model.data<-as.data.frame(cbind(output,mu.cpca.f2.it4.score30))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,mu.cpca.f2.it4.test.score30)
mu.cpca.f2.it4.30.rmse.vec[i]<-
sqrt(mean((svr.pred-(mu.data[mu.sample4,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(mu.cpca.f2.it4.30.rmse.vec,ncol=2,byrow=T))^2))


#Fold 1, Iteration 5
ptm <- proc.time()
mu.cpca.f1.it5=prcomp(image.data[mu.sample5,],center=T,scale=T)
proc.time() - ptm
summary(mu.cpca.f1.it5)

screeplot(mu.cpca.f1.it5, type="l",npcs=50)
abline(h=0,col="blue")
length(which((mu.cpca.f1.it5$sdev)^2>=1))

# 326 PC's under eigenvalues
# 29 PC's under Scree-plot

#Extracting loadings and score matrices
mu.cpca.f1.it5.loading326=mu.cpca.f1.it5$rotation[,1:326]
mu.cpca.f1.it5.loading29=mu.cpca.f1.it5$rotation[,1:29]
mu.cpca.f1.it5.score326=mu.cpca.f1.it5$x[,1:326]
mu.cpca.f1.it5.score29=mu.cpca.f1.it5$x[,1:29]

#Transforming test data into pca-scores
mu.cpca.f1.it5.test.score326=
as.matrix(scale(image.data[-mu.sample5,],center=T,scale=T))%*%
mu.cpca.f1.it5.loading326
mu.cpca.f1.it5.test.score29=
as.matrix(scale(image.data[-mu.sample5,],center=T,scale=T))%*%
mu.cpca.f1.it5.loading29

#Support Vector Regression Models with overal RMSE Estimates
mu.cpca.f1.it5.326.rmse.vec=vector(length=30)

for (i in 1:30){
output<-mu.data[mu.sample5,i]
model.data<-as.data.frame(cbind(output,mu.cpca.f1.it5.score326))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,mu.cpca.f1.it5.test.score326)
mu.cpca.f1.it5.326.rmse.vec[i]<-
sqrt(mean((svr.pred-(mu.data[-mu.sample5,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(mu.cpca.f1.it5.326.rmse.vec,ncol=2,byrow=T))^2))

mu.cpca.f1.it5.29.rmse.vec=vector(length=30)
for (i in 1:30){
output<-mu.data[mu.sample5,i]
model.data<-as.data.frame(cbind(output,mu.cpca.f1.it5.score29))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,mu.cpca.f1.it5.test.score29)
mu.cpca.f1.it5.29.rmse.vec[i]<-
sqrt(mean((svr.pred-(mu.data[-mu.sample5,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(mu.cpca.f1.it5.29.rmse.vec,ncol=2,byrow=T))^2))



#Fold 2, Iteration 5
ptm <- proc.time()
mu.cpca.f2.it5=prcomp(image.data[-mu.sample5,],center=T,scale=T)
proc.time() - ptm
summary(mu.cpca.f2.it5)

screeplot(mu.cpca.f2.it5, type="l",npcs=50)
abline(h=0,col="blue")
length(which((mu.cpca.f2.it5$sdev)^2>=1))

# 328 PC's under eigenvalues
# 30 PC's under Scree-plot

#Extracting loadings and score matrices
mu.cpca.f2.it5.loading328=mu.cpca.f2.it5$rotation[,1:328]
mu.cpca.f2.it5.loading30=mu.cpca.f2.it5$rotation[,1:30]
mu.cpca.f2.it5.score328=mu.cpca.f2.it5$x[,1:328]
mu.cpca.f2.it5.score30=mu.cpca.f2.it5$x[,1:30]

#Transforming test data into pca-scores
mu.cpca.f2.it5.test.score328=
as.matrix(scale(image.data[mu.sample5,],center=T,scale=T))%*%
mu.cpca.f2.it5.loading328
mu.cpca.f2.it5.test.score30=
as.matrix(scale(image.data[mu.sample5,],center=T,scale=T))%*%
mu.cpca.f2.it5.loading30

#Support Vector Regression Models with overal RMSE Estimates

mu.cpca.f2.it5.328.rmse.vec=vector(length=30)
for (i in 1:30){
output<-mu.data[-mu.sample5,i]
model.data<-as.data.frame(cbind(output,mu.cpca.f2.it5.score328))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,mu.cpca.f2.it5.test.score328)
mu.cpca.f2.it5.328.rmse.vec[i]<-
sqrt(mean((svr.pred-(mu.data[mu.sample5,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(mu.cpca.f2.it5.328.rmse.vec,ncol=2,byrow=T))^2))

mu.cpca.f2.it5.30.rmse.vec=vector(length=30)
for (i in 1:30){
output<-mu.data[-mu.sample5,i]
model.data<-as.data.frame(cbind(output,mu.cpca.f2.it5.score30))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,mu.cpca.f2.it5.test.score30)
mu.cpca.f2.it5.30.rmse.vec[i]<-
sqrt(mean((svr.pred-(mu.data[mu.sample5,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(mu.cpca.f2.it5.30.rmse.vec,ncol=2,byrow=T))^2))



#Hubert's PCA with Mean Imputed dataset

#Fold 1, Iteration 1
ptm <- proc.time()
mu.hpca.f1.it1=
PcaHubert(image.data[mu.sample1,],center=T,scale=T,kmax=9216)
proc.time() - ptm
summary(mu.hpca.f1.it1)

screeplot(mu.hpca.f1.it1, type="l",npcs=26,main="mu.hpca.f1.it1")
abline(h=0,col="blue")

# 26 PC's under all
# 16 PC's under Scree-plot

#Extracting loadings and score matrices
mu.hpca.f1.it1.loading26=mu.hpca.f1.it1@loadings[,1:26]
mu.hpca.f1.it1.loading16=mu.hpca.f1.it1@loadings[,1:16]
mu.hpca.f1.it1.score26=mu.hpca.f1.it1@scores[,1:26]
mu.hpca.f1.it1.score16=mu.hpca.f1.it1@scores[,1:16]

#Transforming test data into pca-scores
mu.hpca.f1.it1.test.score26=
as.matrix(scale(image.data[-mu.sample1,],center=T,scale=T))%*%
mu.hpca.f1.it1.loading26
mu.hpca.f1.it1.test.score16=
as.matrix(scale(image.data[-mu.sample1,],center=T,scale=T))%*%
mu.hpca.f1.it1.loading16

#Support Vector Regression Models with overal RMSE Estimates
mu.hpca.f1.it1.26.rmse.vec=vector(length=30)

for (i in 1:30){
output<-mu.data[mu.sample1,i]
model.data<-as.data.frame(cbind(output,mu.hpca.f1.it1.score26))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,mu.hpca.f1.it1.test.score26)
mu.hpca.f1.it1.26.rmse.vec[i]<-
sqrt(mean((svr.pred-(mu.data[-mu.sample1,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(mu.hpca.f1.it1.26.rmse.vec,ncol=2,byrow=T))^2))


mu.hpca.f1.it1.16.rmse.vec=vector(length=30)

for (i in 1:30){
output<-mu.data[mu.sample1,i]
model.data<-as.data.frame(cbind(output,mu.hpca.f1.it1.score16))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,mu.hpca.f1.it1.test.score16)
mu.hpca.f1.it1.16.rmse.vec[i]<-
sqrt(mean((svr.pred-(mu.data[-mu.sample1,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(mu.hpca.f1.it1.16.rmse.vec,ncol=2,byrow=T))^2))



#Fold 2 , Iteration 1
ptm <- proc.time()
mu.hpca.f2.it1=
PcaHubert(image.data[-mu.sample1,],center=T,scale=T,kmax=9216)
proc.time() - ptm
summary(mu.hpca.f2.it1)

screeplot(mu.hpca.f2.it1, type="l",npcs=26,main="mu.hpca.f2.it1")
abline(h=0,col="blue")

# 26 PC's under all
# 16 PC's under Scree-plot

#Extracting loadings and score matrices
mu.hpca.f2.it1.loading26=mu.hpca.f2.it1@loadings[,1:26]
mu.hpca.f2.it1.loading16=mu.hpca.f2.it1@loadings[,1:16]
mu.hpca.f2.it1.score26=mu.hpca.f2.it1@scores[,1:26]
mu.hpca.f2.it1.score16=mu.hpca.f2.it1@scores[,1:16]

#Transforming test data into pca-scores
mu.hpca.f2.it1.test.score26=
as.matrix(scale(image.data[mu.sample1,],center=T,scale=T))%*%
mu.hpca.f2.it1.loading26
mu.hpca.f2.it1.test.score16=
as.matrix(scale(image.data[mu.sample1,],center=T,scale=T))%*%
mu.hpca.f2.it1.loading16

#Support Vector Regression Models with overal RMSE Estimates
mu.hpca.f2.it1.26.rmse.vec=vector(length=30)

for (i in 1:30){
output<-mu.data[-mu.sample1,i]
model.data<-as.data.frame(cbind(output,mu.hpca.f2.it1.score26))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,mu.hpca.f2.it1.test.score26)
mu.hpca.f2.it1.26.rmse.vec[i]<-
sqrt(mean((svr.pred-(mu.data[mu.sample1,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(mu.hpca.f2.it1.26.rmse.vec,ncol=2,byrow=T))^2))


mu.hpca.f2.it1.16.rmse.vec=vector(length=30)

for (i in 1:30){
output<-mu.data[-mu.sample1,i]
model.data<-as.data.frame(cbind(output,mu.hpca.f2.it1.score16))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,mu.hpca.f2.it1.test.score16)
mu.hpca.f2.it1.16.rmse.vec[i]<-
sqrt(mean((svr.pred-(mu.data[mu.sample1,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(mu.hpca.f2.it1.16.rmse.vec,ncol=2,byrow=T))^2))



#Fold 1, Iteration 2
ptm <- proc.time()
mu.hpca.f1.it2=
PcaHubert(image.data[mu.sample2,],center=T,scale=T,kmax=9216)
proc.time() - ptm
summary(mu.hpca.f1.it2)

screeplot(mu.hpca.f1.it2, type="l",npcs=26, main="mu.hpca.f1.it2")
abline(h=0,col="blue")

# 26 PC's under all
# 16 PC's under Scree-plot

#Extracting loadings and score matrices
mu.hpca.f1.it2.loading26=mu.hpca.f1.it2@loadings[,1:26]
mu.hpca.f1.it2.loading16=mu.hpca.f1.it2@loadings[,1:16]
mu.hpca.f1.it2.score26=mu.hpca.f1.it2@scores[,1:26]
mu.hpca.f1.it2.score16=mu.hpca.f1.it2@scores[,1:16]

#Transforming test data into pca-scores
mu.hpca.f1.it2.test.score26=
as.matrix(scale(image.data[-mu.sample2,],center=T,scale=T))%*%
mu.hpca.f1.it2.loading26
mu.hpca.f1.it2.test.score16=
as.matrix(scale(image.data[-mu.sample2,],center=T,scale=T))%*%
mu.hpca.f1.it2.loading16

#Support Vector Regression Models with overal RMSE Estimates
mu.hpca.f1.it2.26.rmse.vec=vector(length=30)

for (i in 1:30){
output<-mu.data[mu.sample2,i]
model.data<-as.data.frame(cbind(output,mu.hpca.f1.it2.score26))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,mu.hpca.f1.it2.test.score26)
mu.hpca.f1.it2.26.rmse.vec[i]<-
sqrt(mean((svr.pred-(mu.data[-mu.sample2,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(mu.hpca.f1.it2.26.rmse.vec,ncol=2,byrow=T))^2))


mu.hpca.f1.it2.16.rmse.vec=vector(length=30)

for (i in 1:30){
output<-mu.data[mu.sample2,i]
model.data<-as.data.frame(cbind(output,mu.hpca.f1.it2.score16))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,mu.hpca.f1.it2.test.score16)
mu.hpca.f1.it2.16.rmse.vec[i]<-
sqrt(mean((svr.pred-(mu.data[-mu.sample2,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(mu.hpca.f1.it2.16.rmse.vec,ncol=2,byrow=T))^2))



#Fold 2, Iteration 2
ptm <- proc.time()
mu.hpca.f2.it2=
PcaHubert(image.data[-mu.sample2,],center=T,scale=T,kmax=9216)
proc.time() - ptm
summary(mu.hpca.f2.it2)

screeplot(mu.hpca.f2.it2, type="l",npcs=26, main="mu.hpca.f2.it2")
abline(h=0,col="blue")

# 26 PC's under all
# 16 PC's under Scree-plot

#Extracting loadings and score matrices
mu.hpca.f2.it2.loading26=mu.hpca.f2.it2@loadings[,1:26]
mu.hpca.f2.it2.loading16=mu.hpca.f2.it2@loadings[,1:16]
mu.hpca.f2.it2.score26=mu.hpca.f2.it2@scores[,1:26]
mu.hpca.f2.it2.score16=mu.hpca.f2.it2@scores[,1:16]

#Transforming test data into pca-scores
mu.hpca.f2.it2.test.score26=
as.matrix(scale(image.data[mu.sample2,],center=T,scale=T))%*%
mu.hpca.f2.it2.loading26
mu.hpca.f2.it2.test.score16=
as.matrix(scale(image.data[mu.sample2,],center=T,scale=T))%*%
mu.hpca.f2.it2.loading16

#Support Vector Regression Models with overal RMSE Estimates
mu.hpca.f2.it2.26.rmse.vec=vector(length=30)

for (i in 1:30){
output<-mu.data[-mu.sample2,i]
model.data<-as.data.frame(cbind(output,mu.hpca.f2.it2.score26))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,mu.hpca.f2.it2.test.score26)
mu.hpca.f2.it2.26.rmse.vec[i]<-
sqrt(mean((svr.pred-(mu.data[mu.sample2,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(mu.hpca.f2.it2.26.rmse.vec,ncol=2,byrow=T))^2))


mu.hpca.f2.it2.16.rmse.vec=vector(length=30)

for (i in 1:30){
output<-mu.data[-mu.sample2,i]
model.data<-as.data.frame(cbind(output,mu.hpca.f2.it2.score16))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,mu.hpca.f2.it2.test.score16)
mu.hpca.f2.it2.16.rmse.vec[i]<-
sqrt(mean((svr.pred-(mu.data[mu.sample2,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(mu.hpca.f2.it2.16.rmse.vec,ncol=2,byrow=T))^2))



#Fold 1, Iteration 3
ptm <- proc.time()
mu.hpca.f1.it3=
PcaHubert(image.data[mu.sample3,],center=T,scale=T,kmax=9216)
proc.time() - ptm
summary(mu.hpca.f1.it3)

screeplot(mu.hpca.f1.it3, type="l",npcs=25, main="mu.hpca.f1.it3")
abline(h=0,col="blue")

# 25 PC's under all
# 16 PC's under Scree-plot

#Extracting loadings and score matrices
mu.hpca.f1.it3.loading25=mu.hpca.f1.it3@loadings[,1:25]
mu.hpca.f1.it3.loading16=mu.hpca.f1.it3@loadings[,1:16]
mu.hpca.f1.it3.score25=mu.hpca.f1.it3@scores[,1:25]
mu.hpca.f1.it3.score16=mu.hpca.f1.it3@scores[,1:16]

#Transforming test data into pca-scores
mu.hpca.f1.it3.test.score25=
as.matrix(scale(image.data[-mu.sample3,],center=T,scale=T))%*%
mu.hpca.f1.it3.loading25
mu.hpca.f1.it3.test.score16=
as.matrix(scale(image.data[-mu.sample3,],center=T,scale=T))%*%
mu.hpca.f1.it3.loading16

#Support Vector Regression Models with overal RMSE Estimates
mu.hpca.f1.it3.25.rmse.vec=vector(length=30)

for (i in 1:30){
output<-mu.data[mu.sample3,i]
model.data<-as.data.frame(cbind(output,mu.hpca.f1.it3.score25))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,mu.hpca.f1.it3.test.score25)
mu.hpca.f1.it3.25.rmse.vec[i]<-
sqrt(mean((svr.pred-(mu.data[-mu.sample3,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(mu.hpca.f1.it3.25.rmse.vec,ncol=2,byrow=T))^2))

mu.hpca.f1.it3.16.rmse.vec=vector(length=30)

for (i in 1:30){
output<-mu.data[mu.sample3,i]
model.data<-as.data.frame(cbind(output,mu.hpca.f1.it3.score16))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,mu.hpca.f1.it3.test.score16)
mu.hpca.f1.it3.16.rmse.vec[i]<-
sqrt(mean((svr.pred-(mu.data[-mu.sample3,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(mu.hpca.f1.it3.16.rmse.vec,ncol=2,byrow=T))^2))



#Fold 2, Iteration 3
ptm <- proc.time()
mu.hpca.f2.it3=
PcaHubert(image.data[-mu.sample3,],center=T,scale=T,kmax=9216)
proc.time() - ptm
summary(mu.hpca.f2.it3)

screeplot(mu.hpca.f2.it3, type="l",npcs=26, main="mu.hpca.f2.it3")
abline(h=0,col="blue")

# 26 PC's under all
# 16 PC's under Scree-plot

#Extracting loadings and score matrices
mu.hpca.f2.it3.loading26=mu.hpca.f2.it3@loadings[,1:26]
mu.hpca.f2.it3.loading16=mu.hpca.f2.it3@loadings[,1:16]
mu.hpca.f2.it3.score26=mu.hpca.f2.it3@scores[,1:26]
mu.hpca.f2.it3.score16=mu.hpca.f2.it3@scores[,1:16]

#Transforming test data into pca-scores
mu.hpca.f2.it3.test.score26=
as.matrix(scale(image.data[mu.sample3,],center=T,scale=T))%*%
mu.hpca.f2.it3.loading26
mu.hpca.f2.it3.test.score16=
as.matrix(scale(image.data[mu.sample3,],center=T,scale=T))%*%
mu.hpca.f2.it3.loading16

#Support Vector Regression Models with overal RMSE Estimates

mu.hpca.f2.it3.26.rmse.vec=vector(length=30)
for (i in 1:30){
output<-mu.data[-mu.sample3,i]
model.data<-as.data.frame(cbind(output,mu.hpca.f2.it3.score26))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,mu.hpca.f2.it3.test.score26)
mu.hpca.f2.it3.26.rmse.vec[i]<-
sqrt(mean((svr.pred-(mu.data[mu.sample3,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(mu.hpca.f2.it3.26.rmse.vec,ncol=2,byrow=T))^2))


mu.hpca.f2.it3.16.rmse.vec=vector(length=30)
for (i in 1:30){
output<-mu.data[-mu.sample3,i]
model.data<-as.data.frame(cbind(output,mu.hpca.f2.it3.score16))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,mu.hpca.f2.it3.test.score16)
mu.hpca.f2.it3.16.rmse.vec[i]<-
sqrt(mean((svr.pred-(mu.data[mu.sample3,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(mu.hpca.f2.it3.16.rmse.vec,ncol=2,byrow=T))^2))



#Fold 1, Iteration 4
ptm <- proc.time()
mu.hpca.f1.it4=
PcaHubert(image.data[mu.sample4,],center=T,scale=T,kmax=9216)
proc.time() - ptm
summary(mu.hpca.f1.it4)

screeplot(mu.hpca.f1.it4, type="l",npcs=25,main="mu.hpca.f1.it4")
abline(h=0,col="blue")

# 25 PC's under all
# 16 PC's under Scree-plot

#Extracting loadings and score matrices
mu.hpca.f1.it4.loading25=mu.hpca.f1.it4@loadings[,1:25]
mu.hpca.f1.it4.loading16=mu.hpca.f1.it4@loadings[,1:16]
mu.hpca.f1.it4.score25=mu.hpca.f1.it4@scores[,1:25]
mu.hpca.f1.it4.score16=mu.hpca.f1.it4@scores[,1:16]

#Transforming test data into pca-scores
mu.hpca.f1.it4.test.score25=
as.matrix(scale(image.data[-mu.sample4,],center=T,scale=T))%*%
mu.hpca.f1.it4.loading25
mu.hpca.f1.it4.test.score16=
as.matrix(scale(image.data[-mu.sample4,],center=T,scale=T))%*%
mu.hpca.f1.it4.loading16

#Support Vector Regression Models with overal RMSE Estimates
mu.hpca.f1.it4.25.rmse.vec=vector(length=30)

for (i in 1:30){
output<-mu.data[mu.sample4,i]
model.data<-as.data.frame(cbind(output,mu.hpca.f1.it4.score25))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,mu.hpca.f1.it4.test.score25)
mu.hpca.f1.it4.25.rmse.vec[i]<-
sqrt(mean((svr.pred-(mu.data[-mu.sample4,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(mu.hpca.f1.it4.25.rmse.vec,ncol=2,byrow=T))^2))


mu.hpca.f1.it4.16.rmse.vec=vector(length=30)

for (i in 1:30){
output<-mu.data[mu.sample4,i]
model.data<-as.data.frame(cbind(output,mu.hpca.f1.it4.score16))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,mu.hpca.f1.it4.test.score16)
mu.hpca.f1.it4.16.rmse.vec[i]<-
sqrt(mean((svr.pred-(mu.data[-mu.sample4,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(mu.hpca.f1.it4.16.rmse.vec,ncol=2,byrow=T))^2))



#Fold 2, Iteration 4
ptm <- proc.time()
mu.hpca.f2.it4=
PcaHubert(image.data[-mu.sample4,],center=T,scale=T,kmax=9216)
proc.time() - ptm
summary(mu.hpca.f2.it4)

screeplot(mu.hpca.f2.it4, type="l",npcs=26, main="mu.hpca.f2.it4")
abline(h=0,col="blue")

# 26 PC's under all
# 16 PC's under Scree-plot

#Extracting loadings and score matrices
mu.hpca.f2.it4.loading26=mu.hpca.f2.it4@loadings[,1:26]
mu.hpca.f2.it4.loading16=mu.hpca.f2.it4@loadings[,1:16]
mu.hpca.f2.it4.score26=mu.hpca.f2.it4@scores[,1:26]
mu.hpca.f2.it4.score16=mu.hpca.f2.it4@scores[,1:16]

#Transforming test data into pca-scores
mu.hpca.f2.it4.test.score26=
as.matrix(scale(image.data[mu.sample4,],center=T,scale=T))%*%
mu.hpca.f2.it4.loading26
mu.hpca.f2.it4.test.score16=
as.matrix(scale(image.data[mu.sample4,],center=T,scale=T))%*%
mu.hpca.f2.it4.loading16

#Support Vector Regression Models with overal RMSE Estimates

mu.hpca.f2.it4.26.rmse.vec=vector(length=30)
for (i in 1:30){
output<-mu.data[-mu.sample4,i]
model.data<-as.data.frame(cbind(output,mu.hpca.f2.it4.score26))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,mu.hpca.f2.it4.test.score26)
mu.hpca.f2.it4.26.rmse.vec[i]<-
sqrt(mean((svr.pred-(mu.data[mu.sample4,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(mu.hpca.f2.it4.26.rmse.vec,ncol=2,byrow=T))^2))


mu.hpca.f2.it4.16.rmse.vec=vector(length=30)
for (i in 1:30){
output<-mu.data[-mu.sample4,i]
model.data<-as.data.frame(cbind(output,mu.hpca.f2.it4.score16))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,mu.hpca.f2.it4.test.score16)
mu.hpca.f2.it4.16.rmse.vec[i]<-
sqrt(mean((svr.pred-(mu.data[mu.sample4,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(mu.hpca.f2.it4.16.rmse.vec,ncol=2,byrow=T))^2))



#Fold 1, Iteration 5
ptm <- proc.time()
mu.hpca.f1.it5=
PcaHubert(image.data[mu.sample5,],center=T,scale=T,kmax=9216)
proc.time() - ptm
summary(mu.hpca.f1.it5)

screeplot(mu.hpca.f1.it5, type="l",npcs=25,main="mu.hpca.f1.it5")
abline(h=0,col="blue")

# 25 PC's under all
# 16 PC's under Scree-plot

#Extracting loadings and score matrices
mu.hpca.f1.it5.loading25=mu.hpca.f1.it5@loadings[,1:25]
mu.hpca.f1.it5.loading16=mu.hpca.f1.it5@loadings[,1:16]
mu.hpca.f1.it5.score25=mu.hpca.f1.it5@scores[,1:25]
mu.hpca.f1.it5.score16=mu.hpca.f1.it5@scores[,1:16]

#Transforming test data into pca-scores
mu.hpca.f1.it5.test.score25=
as.matrix(scale(image.data[-mu.sample5,],center=T,scale=T))%*%
mu.hpca.f1.it5.loading25
mu.hpca.f1.it5.test.score16=
as.matrix(scale(image.data[-mu.sample5,],center=T,scale=T))%*%
mu.hpca.f1.it5.loading16

#Support Vector Regression Models with overal RMSE Estimates
mu.hpca.f1.it5.25.rmse.vec=vector(length=30)

for (i in 1:30){
output<-mu.data[mu.sample5,i]
model.data<-as.data.frame(cbind(output,mu.hpca.f1.it5.score25))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,mu.hpca.f1.it5.test.score25)
mu.hpca.f1.it5.25.rmse.vec[i]<-
sqrt(mean((svr.pred-(mu.data[-mu.sample5,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(mu.hpca.f1.it5.25.rmse.vec,ncol=2,byrow=T))^2))


mu.hpca.f1.it5.16.rmse.vec=vector(length=30)

for (i in 1:30){
output<-mu.data[mu.sample5,i]
model.data<-as.data.frame(cbind(output,mu.hpca.f1.it5.score16))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,mu.hpca.f1.it5.test.score16)
mu.hpca.f1.it5.16.rmse.vec[i]<-
sqrt(mean((svr.pred-(mu.data[-mu.sample5,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(mu.hpca.f1.it5.16.rmse.vec,ncol=2,byrow=T))^2))



#Fold 2, Iteration 5
ptm <- proc.time()
mu.hpca.f2.it5=
PcaHubert(image.data[-mu.sample5,],center=T,scale=T,kmax=9216)
proc.time() - ptm
summary(mu.hpca.f2.it5)

screeplot(mu.hpca.f2.it5, type="l",npcs=26, main="mu.hpca.f2.it5")
abline(h=0,col="blue")

# 26 PC's under all
# 16 PC's under Scree-plot

#Extracting loadings and score matrices
mu.hpca.f2.it5.loading26=mu.hpca.f2.it5@loadings[,1:26]
mu.hpca.f2.it5.loading16=mu.hpca.f2.it5@loadings[,1:16]
mu.hpca.f2.it5.score26=mu.hpca.f2.it5@scores[,1:26]
mu.hpca.f2.it5.score16=mu.hpca.f2.it5@scores[,1:16]

#Transforming test data into pca-scores
mu.hpca.f2.it5.test.score26=
as.matrix(scale(image.data[mu.sample5,],center=T,scale=T))%*%
mu.hpca.f2.it5.loading26
mu.hpca.f2.it5.test.score16=
as.matrix(scale(image.data[mu.sample5,],center=T,scale=T))%*%
mu.hpca.f2.it5.loading16

#Support Vector Regression Models with overal RMSE Estimates
mu.hpca.f2.it5.26.rmse.vec=vector(length=30)

for (i in 1:30){
output<-mu.data[-mu.sample5,i]
model.data<-as.data.frame(cbind(output,mu.hpca.f2.it5.score26))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,mu.hpca.f2.it5.test.score26)
mu.hpca.f2.it5.26.rmse.vec[i]<-
sqrt(mean((svr.pred-(mu.data[mu.sample5,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(mu.hpca.f2.it5.26.rmse.vec,ncol=2,byrow=T))^2))


mu.hpca.f2.it5.16.rmse.vec=vector(length=30)

for (i in 1:30){
output<-mu.data[-mu.sample5,i]
model.data<-as.data.frame(cbind(output,mu.hpca.f2.it5.score16))
svr.model<-svm(output~., data=model.data)
svr.pred<-predict(svr.model,mu.hpca.f2.it5.test.score16)
mu.hpca.f2.it5.16.rmse.vec[i]<-
sqrt(mean((svr.pred-(mu.data[mu.sample5,i]))^2))
}
sqrt(mean(rowMeans
	(matrix(mu.hpca.f2.it5.16.rmse.vec,ncol=2,byrow=T))^2))

#Size of every object in the workspace
size = 0
for (x in ls() ){
    thisSize = object.size(get(x))
    size = size + thisSize
    message(x, " = ", appendLF = F); print(thisSize, units='auto')
}
