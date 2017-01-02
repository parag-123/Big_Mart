library(plyr)
library(dplyr)
library(mice)
library(VIM)
library(ggplot2)
library(lattice) #for density plots
library(corrplot)
library(ggbiplot)
library(rgl)
library(gridExtra)
library(caret)
library(parallel)
library(doParallel)

setwd("D:Your_destination")
train = read.csv("Train.csv")
test = read.csv("Test.csv")
str(big_train)
summary(big_train)
str(big_test)
summary(big_test)

#check dimesions ( number of row & columns) in data sets
dim(big_train)
dim(big_test)

# Check for missing values
table(is.na(big_train))

# Check which rows/col has NA values. It shows Item_weight has ma NA
colSums(is.na(big_train))

summary(big_train)

# Here are some quick inferences drawn from variables in train data set:
#1. Item_Fat_Content has mis-matched factor levels.
#2. Minimum value of item_visibility is 0. Practically, this is not possible. If an item occupies shelf space in a
#grocery store, it ought to have some visibility. We'll treat all 0's as missing values.
#3. Item_Weight has 1463 missing values (already explained above).
#4. Outlet_Size has a unmatched factor levels.
#These inference will help us in treating these variable more accurately.

# The graph shows that many values in item visibility is 0 & max sales which are visiblity is less than 0.2
ggplot(big_train, aes(x= Item_Visibility, y = Item_Outlet_Sales)) + geom_point(size =2.5, color="navy") + xlab("Item Visibility") + ylab("Item Outlet Sales") +
  ggtitle("Item Visibility vs Item Outlet Sales")

# Plotting Outlet identifier & item outlet sales. it shows outlet 7 has max sales
ggplot(big_train, aes(Outlet_Identifier, Item_Outlet_Sales)) + geom_bar(stat ="identity", color = "purple") +
  theme(axis.text.x = element_text(angle = 70, vjust =0.5, color = "black")) + ggtitle("Outlets vs Total Sales") + theme_bw()

# ITem type vs sales
ggplot(big_train, aes(Item_Type, Item_Outlet_Sales)) + geom_bar( stat = "identity")
+theme(axis.text.x = element_text(angle = 70, vjust = 0.5, color = "navy")) +
  xlab("Item Type") + ylab("Item Outlet Sales")+ggtitle("Item Type vs Sales")

#Box plot for same
ggplot(big_train, aes(Item_Type, Item_MRP)) +geom_boxplot() +ggtitle("Box Plot") +
  theme(axis.text.x = element_text(angle = 70, vjust = 0.5, color = "red")) +
  xlab("Item Type") + ylab("Item MRP") + ggtitle("Item Type vs Item MRP")

#Combine dataset

big_test$Item_Outlet_Sales <-  1
combi <- rbind(big_train, big_test)

# Replace NA values with median
combi$Item_Weight[is.na(combi$Item_Weight)] <- median(combi$Item_Weight, na.rm =TRUE)
table(is.na(combi))

#Let's take up Item_Visibility. In the graph above, we saw item visibility has zero value also, which is practically not
#feasible. Hence, we'll consider it as a missing value and once again make the imputation using median.

combi$Item_Visibility <- ifelse(combi$Item_Visibility ==0,median(combi$Item_Visibility),combi$Item_Visibility)


# Correcting mismatched levels in variables outlet size. Replacing other in place of blanks

levels(combi$Outlet_Size)[1] <-"Other"
summary(combi)

# Assignning new levels to item fat content & regularising levels. Using library plyr
# below will assign new levels and combine the sum
combi$Item_Fat_Content <- revalue(combi$Item_Fat_Content,c("LF" = "Low Fat", "reg" = "Regular"))
combi$Item_Fat_Content <- revalue(combi$Item_Fat_Content, c("low fat" = "Low Fat"))
table(combi$Item_Fat_Content)

# Taking a look at outlet identifiers

a <-combi%>% group_by(Outlet_Identifier)%>%tally()
#Giving name to second column of dataframe
names(a)[2] = "Outlet_Count"

combi <- full_join(a, combi, by ="Outlet_Identifier")

# Count of item identifiers
b <-combi%>% group_by(Item_Identifier)%>%tally()
names(b)[2] = "Item_count"
head(b)

combi <- merge(b, combi, by = 'Item_Identifier')

# Find how many years old the outlet is in 2013
c <-combi%>%select(Outlet_Establishment_Year)%>%mutate(Outlet_Year = 2013 -combi$Outlet_Establishment_Year)
combi <- full_join(c,combi)

# Keeping 2 letters of item identifiers
q <-substr(combi$Item_Identifier,1,2)
q <-gsub("FD","Food",q)
q <-gsub("DR","Drinks",q)
q <- gsub("NC","Non-Consumable",q)
combi$Item_Type_New = q


#Label Encoding, in simple words, is the practice of numerically encoding (replacing) different levels of a categorical
#variables. For example: In our data set, the variable Item_Fat_Content has 2 levels: Low Fat and Regular. So, we'll
#encode Low Fat as 0 and Regular as 1. This will help us convert a factor variable in numeric variable. This can be
#simply done using if else statement in R.

combi$Item_Fat_Content <- ifelse(combi$Item_Fat_Content =="Regular",1,0)

#Sample of hot encoding

sample <- select(combi,Outlet_Location_Type)
demo_sample <- data.frame(model.matrix(~.-1,sample))
head(demo_sample)

# One more method of hot encoding
install.packages("dummies")
library(dummies)

combi <- dummy.data.frame(combi, names =
                            c('Outlet_Size','Outlet_Location_Type','Outlet_Type', 'Item_Type_New'), sep='_')

str(combi)

#Finally, we'll drop the columns which have either been converted using other variables or are identifier variables.
#This can be accomplished using select from dplyr package.

combi <- select(combi, -c(Item_Identifier, Outlet_Identifier, Item_Fat_Content,
                          Outlet_Establishment_Year,Item_Type))

# Free up some memory

gc(verbose = TRUE)

ls(all = TRUE)

rm(list = ls(all = TRUE)) 
rm(q) 


ls(all = TRUE)

gc(verbose = TRUE)


# Dividing the combi data

new_train <-combi[1:nrow(big_train),]
new_test <- combi[-(1:nrow(big_train)),]

# Linear regression
linear_model <- lm(Item_Outlet_Sales ~ ., data = new_train)
summary(linear_model)

# The adjusted r square is very bad, lets see if variables are corelated
corrplot(new_train)
cor(new_train$Outlet_Count, new_train$`Outlet_Type_GroceryStore`)

# Deceision trees

library(rpart)
library(e1071)
install.packages("rpart.plot")
library(rpart.plot)
library(caret)
#setting the tree controlparameters 
fitControl <- trainControl(method = "cv", number =5)
cartGrid <- expand.grid(.cp=(1:50)*0.01)
#decision tree
tree_model <- train(Item_Outlet_Sales ~ ., data = new_train, method = "rpart",trControl = fitControl, tuneGrid = cartGrid)
print(tree_model)


main_tree <- rpart(Item_Outlet_Sales ~ ., data = new_train, control =
                     rpart.control(cp=0.01))

prp(main_tree)
##############################################
# Linear Model

#create a new variable in testfile
test$Item_Outlet_Sales <-1
#combine train and testdata
combi <- rbind(train,test)
#impute missing value inItem_Weight26/34
combi$Item_Weight[is.na(combi$Item_Weight)] <- median(combi$Item_Weight, na.rm =TRUE)
#impute 0 initem_visibility
combi$Item_Visibility <- ifelse(combi$Item_Visibility == 0,median(combi$Item_Visibility), combi$Item_Visibility)
#rename level inOutlet_Size
levels(combi$Outlet_Size)[1] <-"Other"
#rename levels ofItem_Fat_Content
library(plyr)
combi$Item_Fat_Content <- revalue(combi$Item_Fat_Content,c("LF" = "Low Fat", "reg"= "Regular"))
combi$Item_Fat_Content <- revalue(combi$Item_Fat_Content, c("low fat" = "LowFat"))
#create a new column 2013 -Year
combi$Year <- 2013 -combi$Outlet_Establishment_Year
#drop variables not required inmodeling
library(dplyr)
combi <- select(combi, -c(Item_Identifier, Outlet_Identifier,Outlet_Establishment_Year))
#divide dataset
new_train <-combi[1:nrow(train),]
new_test <- combi[-(1:nrow(train)),]
#linear regression
linear_model <- lm(Item_Outlet_Sales ~ ., data =new_train)
summary(linear_model)

# First submissionusing deceision trees

main_predict <- predict(main_tree, newdata = new_test, type ="vector")
sub_file <- data.frame(Item_Identifier = test$Item_Identifier, Outlet_Identifier =test$Outlet_Identifier, Item_Outlet_Sales = main_predict)
write.csv(sub_file,'Decision_tree_sales.csv')

