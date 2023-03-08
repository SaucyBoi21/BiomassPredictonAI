library(tidyverse)
library(shiny)

plant_data <- read.csv("C:\\Users\\saaha\\OneDrive\\Documents\\GitHub\\BiomassPredictonAI\\harvest_bydate.csv")

View(plant_data)

ggplot(data = plant_data) + geom_point(mapping = aes(x = plant_area, y = LDW_g))

runExample("01_hello")
