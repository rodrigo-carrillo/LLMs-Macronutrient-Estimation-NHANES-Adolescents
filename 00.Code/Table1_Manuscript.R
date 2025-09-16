df <- read.csv("C:/Users/rmcarri/OneDrive - Emory University/NHANES/Macronutrients LLMs/Combined_df_ten_shot_day2.csv")
length(unique(df$id))


prop.table(table(df$sex))
summary(df$age)
summary(df$DRxIKCAL); sd(df$DRxIKCAL)
summary(df$DRxIPROT); sd(df$DRxIPROT)
summary(df$DRxICARB); sd(df$DRxICARB)
summary(df$DRxISUGR); sd(df$DRxISUGR)
summary(df$DRxIFIBE); sd(df$DRxIFIBE)
summary(df$DRxITFAT); sd(df$DRxITFAT)
