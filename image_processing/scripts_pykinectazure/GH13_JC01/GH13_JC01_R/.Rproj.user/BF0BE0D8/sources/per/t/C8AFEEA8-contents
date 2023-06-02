library(tidyverse)
library(dplyr)
library(readxl)
library(reshape2)

geo_traits <- read_xlsx("correlations_all_trays.xlsx")

geo_traits$tray_id <- as.factor(geo_traits$tray_id)


model_width<-lm(geo_traits$plant_width_mm~geo_traits$`width _mm`)
width_sum<- summary(model_traits)
width_sum

model_lenght<- lm(geo_traits$plant_height_mm~geo_traits$length_mm)
lenght_sum<- summary(model_lenght)
lenght_sum

................................................................................................

# What is important to notice is the difference between real value (dimensions using a caliper)
#and the value extracted from the camera
geo_traits %>% ggplot(aes(tray_id,error_width_mm,color=tray_id)) +
  geom_boxplot(width = 0.5,outlier.shape = NA)+
  geom_jitter(width = 0.15,alpha = 0.4)+
  theme_bw()+
  labs(title = "Plant Width: Error Distribuion",
       x = 'Tray',
       y = "Width_Error (mm)",
       caption = "RMSE = 9.71 mm" )

geo_traits %>% ggplot(aes(tray_id,error_lenght_mm,color=tray_id)) +
  geom_boxplot(width = 0.5,outlier.shape = NA)+
  geom_jitter(width = 0.15,alpha = 0.4)+
  theme_bw()+
  labs(title = "Plant Width: Error Distribuion",
       x = 'Tray',
       y = "Lenght_Error(mm)",
       caption = "RMSE = 4.43 mm" )
 
geo_traits %>% ggplot(aes(tray_id,error_lenght_mm,color=tray_id)) +
  geom_boxplot(width = 0.5,outlier.shape = NA)+
  geom_jitter(width = 0.15,alpha = 0.4)+
  theme_bw()+
  labs(title = "Plant Width: Error Distribuion",
       x = 'Tray',
       y = "Lenght_Error(mm)",
       caption = "RMSE = 4.43 mm" )


## Reshape dataframe to visualize one box plot per variable: 
geo_mod <- melt(geo_traits, measure.vars = c('error_lenght_mm','error_width_mm'))

geo_mod %>% ggplot(aes(variable,value,color=variable)) +
  geom_boxplot(width = 0.5,outlier.shape = NA)+
  geom_jitter(width = 0.15,alpha = 0.4)+
  theme_bw()+
  labs(title = "Error Distribuion for Image Derived Traits",
       x = 'Geometric Trait',
       y = "Error (mm)",
       caption = " Width RMSE  = 9.71 mm     Lenght RMSE= 4.43 mm" )

