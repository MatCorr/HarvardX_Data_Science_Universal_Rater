################################################################################
# Universal Rater Project - HarvardX - Data Science Capstone - Project 2
################################################################################

# This code is one of the delivered items of the second (and final)
# project in the Data Science Capstone course by HarvardX, which
# was to be chosen by the learners themselves.
#
# Through this code, we will try to create a model that, by learning from
# gaming review excerpts found in Metacritic, tries to predict the score that
# will be given. The concept behind the task is not simply making accurate
# predictions, but trying to create one Universal Rater: a machine learning
# model that, based on the text written by the reviewer (more specifically
# its conveyed feeling and words), suggests a score, eliminating therefore the 
# bias that every reviewer has when grading a game and giving birth to a 
# suggested universal score that captures the "true" level of like,
# dislike, or indifference found in the text.
#
# This code has been divided into 14 steps that can be grouped as:
# Steps 1-5: Preliminary steps of experimental setup and data preparation.
# Steps 6-7: Creation of baseline classifiers.
# Step 8: Execution of feature selection.
# Steps 9-13: Tuning of the model selected through Steps 6-8.
# Step 14: Execution of tuned model over test set and recording of the results.
#
# Given this code works with Document x Term matrices and runs lots of
# experiments, it takes a while to run completely. It is recommended that anyone
# trying to run it do so section by section, not just because of processing
# matters, but because running the code step by step will also make it easier
# to follow what is being done.
#
# The data used in this experiment was originally found in Kaggle:
# https://www.kaggle.com/skateddu/metacritic-critic-games-reviews-20112019
# So thanks to Andrea Cadeddu, the uploader.


###############################################################################
# Step 1: Importing Packages
#
# In this section, used packages are imported. If they are not yet installed, 
# the installations are done automatically.
###############################################################################

if(!require(caret)) install.packages("caret")
if(!require(dplyr)) install.packages("dplyr")
if(!require(readr)) install.packages("dplyr")
if(!require(stringr)) install.packages("stringr")
if(!require(text2vec)) install.packages("text2vec")
if(!require(tidytext)) install.packages("tidytext")
if(!require(xgboost)) install.packages("xgboost")

library(caret)
library(dplyr)
library(readr)
library(stringr)
library(text2vec)
library(tidytext)
library(xgboost)




###############################################################################
# Step 2: Creating Functions
#
# In this sections are the functions that will be used throughout the code.
# They were created either to improve readability or because they were to be
# called repeatedly during the experiment. Attached to each one is an
# explanation of the task they perform.
#
# More functions could have been created here, but I opted to do so only
# for minor tasks in order to make the code of the individual experiments
# more whole.
###############################################################################

# One of the tasks that will be done in the preparation of the dataset will be
# the removal of game titles from review texts. This is done to stop the
# rater from becoming biased due to the game title (as in giving high scores
# to games with generally high scores). Again, we want the score to be given
# based on the words of the text (the general feeling), not the game.
remove_game_name_from_review_text <- function(game_title, review_text) {
  
  # The title of the game is received and has its white spaces replaced by "|".
  # This string is passed to str_remove_all, which eliminates the individual
  # words separated by "|" from the text. This separation is done because
  # it is common for reviewers to refer to games by just a part of their titles.
  game_name_with_separator <- gsub(' ', '|', game_title)
  adjusted_review <- str_remove_all(review_text, game_name_with_separator)
  return(adjusted_review)
}

# This function is used to remove a list of words from the text and will be
# performed as part of data preparation. The review text is split in its spaces
# and if the word is in the list of removals, it is erased.
remove_words_from_text <- function(text) {
  text <- unlist(strsplit(text, " "))
  paste(text[!text %in% words_to_remove], collapse = " ")
}

# In order to select the most valuable features (terms) used by the model to
# make its decision, we will need to pull the "importance" rates out of them.
# This function does so for both models that will be used, the Linear one, which
# measures such importance in "Weights", and the Tree Model, which measures
# such importance in Gain.
obtain_importance_terms <- function(model, vocabulary_model) {
  
  # The importance of all used terms is obtained via the xgb.importance function.
  importance <- xgb.importance(feature_names = vocabulary_model$term, model = model)
  
  # The value of the importance will be given either by the "Weight" or "Gain"
  # column, depending on the model. Here, that column is renamed "value".
  importance <- importance %>% 
    mutate(value = {if("Weight" %in% names(.)) abs(Weight) else Gain})
  
  # The column Feature is renamed to term.
  importance <- importance %>% mutate(term=Feature) %>%
    select(term, value)
  
  # The list of terms x importance is returned.
  importance
}

# This function adjusts scores to the scale we will be using. Essentially,
# two adjustments are made.
adjust_scores <- function(score) {
  
  # Firstly, given our models return scores on a continuous scale that may
  # occasionally go below 0 or above 100, here they are brought to those
  # limits in case they go under or over them.
  score <- ifelse(score > 100, 100, score)
  score <- ifelse(score < 0, 0, score)
  
  # The database has a few scores that do not end in 0 or 5, like 73. Moreover,
  # the models will predict scores that do not end in those numbers either.
  # Since our universal rater will operate only with scores ending in 0 or 5,
  # and since scores that do not end in those numbers are very rare in the
  # database, here they are brought to those values. If a score is 72.4, it will
  # become 70. If it is 72.51, it will become 75, for example.
  if (score %% 5 == 0) {
    score
  } else if (score %% 5 <= 2.5) {
    score - score %% 5
  } else {
    score - score %% 5 + 5
  }
}




###############################################################################
# Step 3: Downloading Data
#
# In this section, data to be used in the experiment is downloaded from the
# GitHub repository where it is located and loaded into a data frame.
###############################################################################

# The data used in the experiment, obtained originally in Kaggle, was made
# available in my GitHub repository. Here, it is downloaded from there.
# I figured it would be a safer place to store it in for this download to work
# because datasets can be removed by the uploader from Kaggle sometimes.
file_address <- "https://raw.githubusercontent.com/MatCorr/HarvardX_Data_Science_Universal_Rater/main/data/metacritic_critic_reviews.csv"

reviews_ds <- read_csv(file_address)

# Removing variables that will not be used from this point forward.
# This will be done at the end of every section to keep the environment as light
# as possible and clean.
rm(file_address)


###############################################################################
# Step 4: Data Wrangling
#
# In this section, data preparation is done. Given this is a dataset strongly
# based on text, we mostly prepare that data to be analyzed by the models.
###############################################################################

# Some reviews did not have scores attached to them. Here, those are removed.
reviews_ds <- reviews_ds[-which(is.na(reviews_ds$score)),]

# Moreover, some reviews had no actual text, containing the "Quotation 
# forthcoming" Metacritic placeholder. Here, they are also taken out.
reviews_ds <- reviews_ds[-which(reviews_ds$review == "Quotation forthcoming."),]

# Some scores do not use 5 increments (60, 65, 70, etc.). Here, they are rounded
# to the closest increment of 5, since that's the scale our rater will use.
reviews_ds$score <- sapply(reviews_ds$score, adjust_scores)

# Creating an ID column for the data set, which lacks one. This will be 
# essential down the line when working with Document x Term matrices.
reviews_ds$reviewId <- seq.int(nrow(reviews_ds))

# The dataset has a few columns that will not be of interest to the work
# being done here, such as review source and game title. Here, we select
# only the columns we will work with. They are loaded into a variable
# called original_reviews so we can retain them, in their mostly unformatted
# format, until the end of the experiment.
original_reviews <- reviews_ds %>% select(reviewId, review, score)

# Removing punctuation from the reviews.
reviews_ds$review <- str_replace_all(reviews_ds$review, "[[:punct:]]", " ")

# Removing punctuation and the symbols + and & from game titles. This is of
# importance to the next operation in the code (the removal of game names from
# reviews). The + and & symbols were eliminated because they were causing an
# error in the removal operation. Thanks, Nintendogs + Cats and others.
reviews_ds$game <- str_replace_all(reviews_ds$game, "[[:punct:]]", " ")
reviews_ds$game <- str_replace_all(reviews_ds$game, "\\+|\\&", " ")

# Here reviews are processed one by one and game titles are removed from them.
# Again, this is done because we want the model to guess the score based on the
# text's feeling, not the name of the game, which is strongly correlated to
# the score.
reviews_ds$review <- mapply(remove_game_name_from_review_text, 
                            reviews_ds$game,
                            reviews_ds$review,
                            SIMPLIFY = TRUE)

# Review text is brought to lower case. 
reviews_ds$review <- tolower(reviews_ds$review)

# Digits are removed later from the reviews because many games have names with
# digits. As such, removing numbers before removing the names of games could
# cause the exclusion of game names from the texts to not work as intended.
reviews_ds$review <- str_replace_all(reviews_ds$review, "[[:digit:]]+", " ")

# Just making sure all white spaces in the text are single spaces. A little
# cleanup to organize things after all this formatting, and before even more
# formatting.
reviews_ds$review <- str_squish(reviews_ds$review)

# Eliminating columns that will not be used.
reviews_ds <- reviews_ds %>% select(reviewId, review, score)




###############################################################################
# Step 5: Dataset Splits and Some Extra Data Preparation
#
# In this section, the full dataset is split into three:
# Train, which will be used to train the models.
# Validation, which will be used to evaluate and refine trained models.
# Test, which will be used only by the best model to generate the final results.
#
# Plus, we will remove words with low frequencies (lower than 4) only from the
# training dataset. Both the counting and the removal are done just with the
# training dataset to avoid data leakage to occur.
###############################################################################

# Setting the seed so the experiment's results can be reproduced
set.seed(1, sample.kind="Rounding")

# Splitting the dataset into train and test. Test will be 10% of the total 
# dataset, that's 12,325 rows, and train will get the remaining 90%. The test 
# dataset will only be used in the final section to yield the final results.
test_index <- createDataPartition(reviews_ds$score, p = .1, list = FALSE)
train <- reviews_ds[-test_index,]
test <- reviews_ds[test_index,]

# Here, we use temp as a temporary variable to store the train dataset so that
# there is no problem in the partitioning of the data via the index.
temp <- train

# The train dataset is further split into train, the data that will in fact
# be used for training, and validation, the data that will be used to check
# the performance of the trained models. 
validation_index <- createDataPartition(temp$score, p = .1, list = FALSE)
validation <- temp[validation_index,]
train <- temp[-validation_index,]

# Here, a final step of data preparation is done. Note that this is done after
# the dataset was split in order to avoid data leakage. Here we will count
# and eliminate words with low frequencies, and we don't want the frequency
# of words in the test dataset to influence that operation. First, we get all 
# the vocabulary in the text and count how frequently each word
# shows up (frequency).
words <- strsplit(train$review, " ")
words <- unlist(words)
word_counts <- data.frame(table(words))

# This conversion is done because the column words is returned as a factor
# from the function above, and we'll need it to be a character for the next
# step.
word_counts$words <- as.character(word_counts$words)

# Here, we obtain words that appear less than four times, and also words with
# only one character (there was some trash left in there after all this
# cleaning).
words_to_remove_index <- which(word_counts$Freq <= 3 | nchar(word_counts$words) == 1)
words_to_remove <- word_counts[words_to_remove_index,]$words

# Next, we eliminate those undesirables. Note that this lapply function took a 
# little bit of time to run on my notebook.
train$review <- lapply(train$review, remove_words_from_text)
train$review <- as.character(train$review)

# A variable with the full vocabulary (of the training set) is kept for future
# use. Not in this code, but in the making of the report that will accompany it.
complete_vocabulary <- word_counts[-words_to_remove_index,]

# Once more, variables that will not be reused are removed.
rm(test_index, validation_index, reviews_ds, temp, words, word_counts,
   words_to_remove_index, words_to_remove)




###############################################################################
# Step 6: Experiment 1.1 - Baselines - Training
#
# In this section, our baseline models are trained. We will try to improve
# on their performance via the next two experiments.
###############################################################################

# Now we are ready to train our first models. Before doing so, some extra text
# preparation has to be done. The first part is to tokenize the text (separete
# it into individual words). Here, the function that will do the tokenizing
# is instanced.
tokenizer_function <- word_tokenizer

# The function tokenizes the text of the training reviews.
train_tokens <- tokenizer_function(train$review)

# Next, we create an iterator that will read these tokens one by one and
# transform them. We also tell it the column that uniquely identifies all
# reviews so it knows which transformed results belong to whom.
iterator_train <- itoken(train_tokens, 
                         ids = train$reviewId,
                         progressbar = FALSE)

# The iterator is used to create a vocabulary with all words that are present in
# all the training reviews.
vocabulary <- create_vocabulary(iterator_train)

# The vocabulary is then fed into a vectorizer. This guy will turn the tokenized
# texts into vectors.
vectorizer <- vocab_vectorizer(vocabulary)

# Here we get our Document x Term matrix. Each review is a row, and each column
# represents a word. Each row x column is filled with the number of times
# the word appears in the review.
document_term_matrix_train <- create_dtm(iterator_train, vectorizer)

# We, however, don't want to simply get the counts of when all words appear
# in all reviews. We want a normalized number that will be influenced how 
# frequent the word is in all texts. TF-IDF will give us that, so we run it.
model_tf_idf <- TfIdf$new()
document_term_matrix_train <- model_tf_idf$fit_transform(document_term_matrix_train)

# Here, we train the first of our baseline models via the library XGBoost.
# It is a linear model, but with a twist, as it uses an extreme boosting
# technique (hence the name) to create a lot of weak models and combine their
# results into a - hopefully - good one. We also set the seed so the experiment
# can be reproduced and yield the same results. The metric we are trying to
# minimize is RMSE.
set.seed(1, sample.kind="Rounding")
linear_model <- xgboost(data = document_term_matrix_train,
                        label = train$score,
                        booster = 'gblinear',
                        nround = 500,
                        objective = "reg:squarederror")

# We get the importance each term had for the linear model. This is indicated
# by the "weight" variable. This importance will be employed in the next
# experiment.
importance_terms_linear <- obtain_importance_terms(linear_model, vocabulary)

# We set the seed again and train our second baseline model. A regression tree
# (actually many regression trees), that are being built via the XGBoost
# technique.
set.seed(1, sample.kind="Rounding")
tree_model <- xgboost(data = document_term_matrix_train,
                      label = train$score,
                      booster = 'gbtree',
                      nround = 500,
                      objective = "reg:squarederror")

# Once again, we get the importance of the terms. In the case of the tree,
# though, that importance is given by the variable Gain. Here, by using the
# function, that difference is transparent. The function takes care of that.
importance_terms_tree <- obtain_importance_terms(tree_model, vocabulary)

# Once more, variables that will not be reused are eliminated.
rm(document_term_matrix_train, iterator_train, train_tokens)




###############################################################################
# Step 7: Experiment 1.2 - Baselines - Validation
#
# In this section, the models that were trained in Section 6 are evaluated
# against the validation dataset. Their overall results, measured in RMSE,
# are written to a data frame.
###############################################################################

# Before we do the validation (that is, applying the models built in the 
# previous section to the validation dataset to see how well they will 
# perform), we need to tokenize/vectorize the text of the validation reviews.
# So the same process done for the train set is executed here.
validation_tokens <- tokenizer_function(validation$review)

iterator_validation <- itoken(validation_tokens, 
                              ids = validation$reviewId, 
                              progressbar = FALSE)

document_term_matrix_validation <- create_dtm(iterator_validation, vectorizer)

# There is one difference, though, and this is it. The TF-IDF model that was fit 
# to our train set is reused. This is done so the rows (reviews) of the validation 
# set have the same columns (words) as the ones in the training set. Otherwise, 
# the models would not work, since the dimensions of the training and validation
# matrices would be different. Plus, reusing this TF-IDF here allows information 
# on the frequency of words in training data to be used to normalize validation 
# data.
document_term_matrix_validation <- model_tf_idf$fit_transform(document_term_matrix_validation)

# The matrix is ready, so we generate the predictions with the linear and
# tree model.
predictions_linear <- predict(linear_model, document_term_matrix_validation)
predictions_tree <- predict(tree_model, document_term_matrix_validation)

# We adjust the generated scores so they are all between 0-100 and also
# respect our scale of 5-point increments. This is done because our models
# generate continuous numbers, and we want our rater to give scores with
# 5-point increments.
predictions_linear <- sapply(predictions_linear, adjust_scores)
predictions_tree <- sapply(predictions_tree, adjust_scores)

# The validation results obtained by our models are recorded into a data frame.
# The data frame is called results_feature_selection because it will continue to
# be used in the next experiment, when feature selection is done. In a sense,
# this is iteration 0 of that experiment, since all features (terms) of the
# training vocabulary are considered.
results_feature_selection <- data.frame(Model='Linear',
                                        iteration=0,
                                        terms_used=nrow(vocabulary), 
                                        rmse=RMSE(predictions_linear, validation$score))

results_feature_selection <- rbind(results_feature_selection , data.frame(Model='Tree',
                                                                          iteration=0,
                                                                          terms_used=nrow(vocabulary),
                                                                          rmse=RMSE(predictions_tree, validation$score)))

# We create a variable called vocabulary_best_model because we want to keep
# track of the words used by our best model for when we train what will be our
# final model. Since so far there have only been two models trained and the two
# of them have the same vocabulary, the variable simply receives that list of
# words.
vocabulary_best_model <- vocabulary

# Again, removing variables we won't need anyone.
rm(document_term_matrix_validation, tokenizer_function, predictions_linear, 
   predictions_tree, linear_model, tree_model, validation_tokens, model_tf_idf,
   vocabulary, vectorizer, iterator_validation)




###############################################################################
# Step 8: Experiment 2 - Feature Selection - Training and Validation
#
# In this section we will see if removing extra features (terms) from the
# vocabulary will improve the models' capacity to predict. 30 iterations will
# be executed and, for each one, the bottom 1% important features, as reported
# by each algorithm, will be ignored when building the document x term matrix.
###############################################################################

# Since the experiment will be run 30 times, each time around with less
# vocabulary being used, a for loop is created.
for (i in 1:30) {
  
  # Just a little console message to keep track of each iteration is being
  # executed. After all, this can take a while.
  cat(paste('Iteration', i, 'of 30 started.\n'))
  
  # Here, we eliminate words from the vocabulary that will be used by the models.
  # Given the two models (trained with all the terms in the previous experiment) 
  # return the importance of the terms for the decision they made, we order 
  # (descending) by those values (Gain for the tree and Weight for 
  # the linear regression). We then calculate how many terms correspond to 1% of 
  # the vocabulary and take them out.
  
  # It is, however, important to note that while the linear model returns
  # weights for all terms used, the tree reports Gain only for terms that
  # were employed in creating the branches. As such, for the trees, the
  # elimination  being done is far more aggressive, as it cuts out not just the
  # bottom 1% but also all terms that were not used.
  
  vocabulary_to_keep_linear <- importance_terms_linear %>% arrange(-value)
  threshold_index <- nrow(vocabulary_to_keep_linear) - (nrow(vocabulary_to_keep_linear)/100)
  vocabulary_to_keep_linear <- vocabulary_to_keep_linear[1:floor(threshold_index),]
  
  vocabulary_to_keep_tree <- importance_terms_tree %>% arrange(-value)
  threshold_index <- nrow(vocabulary_to_keep_tree) - (nrow(vocabulary_to_keep_tree)/100)
  vocabulary_to_keep_tree <- vocabulary_to_keep_tree[1:floor(threshold_index),]
  
  # Let's see how the models do with 1% (or more, in the case of the tree) less
  # vocabulary than the last time around. We are creating new tokenizer and
  # vectorizer functions here. It is the same procedure done in Section 6.
  tokenizer_function <- word_tokenizer
  train_tokens <- tokenizer_function(train$review)
  iterator_train <- itoken(train_tokens, 
                           ids = train$reviewId,
                           progressbar = FALSE)
  vocabulary <- create_vocabulary(iterator_train)
  
  # Here comes the difference. At this point, the vocabulary variable (which 
  # will determine the columns of our Document X Term matrix) has all words
  # of the train dataset (that is, the full vocabulary). We want to cut terms
  # that are in the bottom 1% or, in the case of the tree, those in the
  # bottom 1% plus those that were not used. Here, that exclusion is done.
  vocabulary_linear <- vocabulary[(vocabulary$term %in% vocabulary_to_keep_linear$term), ]
  vocabulary_tree <- vocabulary[(vocabulary$term %in% vocabulary_to_keep_tree$term), ]
  
  # We vectorize with the new, reduced, vocabulary.
  vectorizer_linear <- vocab_vectorizer(vocabulary_linear)
  vectorizer_tree <- vocab_vectorizer(vocabulary_tree)
  
  # We create the Document x Term matrices. Since the vocabularies of the models
  # are, at this point, different, we need two of everything.
  document_term_matrix_train_linear <- create_dtm(iterator_train, vectorizer_linear)
  document_term_matrix_train_tree <- create_dtm(iterator_train, vectorizer_tree)
  
  # For example, we also need two TF-IDF models, because the normalization will
  # be different.
  model_tf_idf_linear <- TfIdf$new()
  document_term_matrix_train_linear <- model_tf_idf_linear$fit_transform(document_term_matrix_train_linear)
  
  model_tf_idf_tree <- TfIdf$new()
  document_term_matrix_train_tree <- model_tf_idf_tree$fit_transform(document_term_matrix_train_tree)
  
  # Now we train the models, but not before setting the seed so the experiment 
  # can be reproduced with the same results as those reported. Note that, since
  # we want to keep reducing the vocabulary with every iteration to see how the
  # models react, we get the importance of the terms from both models again.
  set.seed(1, sample.kind="Rounding")
  linear_model <- xgboost(data = document_term_matrix_train_linear,
                          label = train$score,
                          booster = 'gblinear',
                          nround = 500,
                          objective = "reg:squarederror")
  
  importance_terms_linear <- obtain_importance_terms(linear_model, vocabulary_linear)
  
  
  set.seed(1, sample.kind="Rounding")
  tree_model <- xgboost(data = document_term_matrix_train_tree,
                        label = train$score,
                        booster = 'gbtree',
                        nround = 500,
                        objective = "reg:squarederror")
  
  importance_terms_tree <- obtain_importance_terms(tree_model, vocabulary_tree)

  # These Document x Term matrices can be quite heavy. So, before creating two
  # more (the ones that will be used for validation), we remove these.
  rm(document_term_matrix_train_linear, document_term_matrix_train_tree)
  
  
  # Standard procedure here. We are creating the tokenizer and iterator for
  # the validation dataset.
  validation_tokens <- tokenizer_function(validation$review)
  
  iterator_validation <- itoken(validation_tokens, 
                                ids = validation$reviewId, 
                                progressbar = FALSE)
  
  # Here we have to create two Document x Term matrices for validation. The
  # reason is, of course, because our models were trained with different 
  # vocabulary so they are ready to make predictions on matrices with
  # different dimensions. TF-IDF is also executed separately.
  document_term_matrix_validation_linear <- create_dtm(iterator_validation, vectorizer_linear)
  document_term_matrix_validation_linear <- model_tf_idf_linear$fit_transform(document_term_matrix_validation_linear)
  
  document_term_matrix_validation_tree <- create_dtm(iterator_validation, vectorizer_tree)
  document_term_matrix_validation_tree <- model_tf_idf_tree$fit_transform(document_term_matrix_validation_tree)
  
  # The predictions are generated.
  predictions_linear <- predict(linear_model, document_term_matrix_validation_linear)
  predictions_tree <- predict(tree_model, document_term_matrix_validation_tree)
  
  # The scores are adjusted so that they are in our previously explained final 
  # output scale.
  predictions_linear <- sapply(predictions_linear, adjust_scores)
  predictions_tree <- sapply(predictions_tree, adjust_scores)
  
  # We calculate RMSE between the predictions and the actual scores of the
  # validation dataset.
  rmse_linear <- RMSE(predictions_linear, validation$score)
  rmse_tree <- RMSE(predictions_tree, validation$score)
  
  # Remember, in order to build our final model we will need to know which
  # vocabulary it used. So we will need to keep track of which model is doing
  # the best so far. Here, we get the current best score from the results data 
  # frame. In the first iteration that result will, naturally, be one of the
  # two calculated in Experiment 1.
  current_best <- min(results_feature_selection$rmse)
  
  # The results (RMSE) for our models with reduced vocabulary are added to the
  # data frame. We keep track of the iteration in which they were produced and
  # how many words were present in their vocabulary.
  results_feature_selection <- rbind(results_feature_selection , data.frame(Model='Linear',
                                                                            iteration=i,
                                                                            terms_used=nrow(vocabulary_linear),
                                                                            rmse=RMSE(predictions_linear, validation$score)))
  
  results_feature_selection <- rbind(results_feature_selection , data.frame(Model='Tree',
                                                                            iteration=i,
                                                                            terms_used=nrow(vocabulary_tree),
                                                                            rmse=RMSE(predictions_tree, validation$score)))
  
  # Getting the name of the winner of this iteration.
  iteration_winner <- ifelse(rmse_linear < rmse_tree, 'Linear', 'Tree')
  
  # Did any of our new models with reduced vocabulary beat the current best?
  # If that's the case, the current_best variable is updated and the vocabulary
  # of the new best model is saved.
  if (min(rmse_linear, rmse_tree) < current_best & iteration_winner == 'Linear') {
    vocabulary_best_model <- vocabulary_linear
    current_best <- min(rmse_linear, rmse_tree)
  } else if (min(rmse_linear, rmse_tree) < current_best & iteration_winner == 'Tree') {
    vocabulary_best_model <- vocabulary_tree
    current_best <- min(rmse_linear, rmse_tree)
  }
  
  # Removing objects that will be recreated in the next iteration of the 
  # experiment.
  rm(document_term_matrix_validation_linear, document_term_matrix_validation_tree,
     vectorizer_linear, vectorizer_tree, model_tf_idf_linear, model_tf_idf_tree,
     validation_tokens, iterator_validation, train_tokens, iterator_train, 
     linear_model, tree_model)
  
  # Just another message to keep track of iteration numbers.
  cat(paste('Iteration', i, 'of 30 executed.\n')) 
}

# After all this playing with vocabulary and division between linear and tree,
# there are many variables to eliminate.
rm(vocabulary_linear, vocabulary_tree, vocabulary_to_keep_linear, vocabulary_to_keep_tree, 
   predictions_linear, predictions_tree, iteration_winner, rmse_linear, rmse_tree,
   threshold_index, current_best, importance_terms_linear, importance_terms_tree, i)




###############################################################################
# Step 9: Experiment 3.1 - Tuning - Nround
#
# In this Experiment, we work on the model we settled on during the previous 
# experiment: the XGBoost Tree with 2,603 terms. We will try to tune
# its parameters to further improve its performance. In this step, 
# we look for the best value of nround.
#
# IMPORTANT NOTE: Using grid tuning from caret would have made this section
# a lot cleaner in terms of code, but then I would have had to turn the
# Document x Term Matrix into a sparse matrix and my notebook couldn't handle
# it.
###############################################################################

# The values of nround which will be tested.
nround_values <- c(50, 100, 150, 200, 250, 300, 350, 400, 450, 500)

# As it happened with the Feature Selection experiment, we are going to
# store the result of each iteration into a data frame. Here, it is
# created.
results_nround_tuning <- data.frame(nround=numeric(),
                                    rmse=numeric()) 

# The number of repetitions of this experiment will be equal to the number of
# items in nround_values.
for (i in 1:length(nround_values)) {
  
  # Control message.
  cat(paste('Iteration', i, 'of', length(nround_values), 'started.\n'))
  
  # The usual procedure once more. Our tokenizer and vectorizer are created.
  tokenizer_function <- word_tokenizer
  train_tokens <- tokenizer_function(train$review)
  iterator_train <- itoken(train_tokens, 
                           ids = train$reviewId,
                           progressbar = FALSE)
  
  # Our vocabulary will be cut down to the words used by the best model from
  # the previous experiment. It is this guy we want to tune as much as possible.
  vocabulary <- create_vocabulary(iterator_train)
  vocabulary <- vocabulary[(vocabulary$term %in% vocabulary_best_model$term), ]
  vectorizer <- vocab_vectorizer(vocabulary)
  
  # We get the Document x Term matrix and apply TF-IDF to it. 
  document_term_matrix_train <- create_dtm(iterator_train, vectorizer)
  model_tf_idf <- TfIdf$new()
  document_term_matrix_train <- model_tf_idf$fit_transform(document_term_matrix_train)
  
  # We train the model with the value of nround determined by the iteration.
  set.seed(1, sample.kind="Rounding")
  model <- xgboost(data = document_term_matrix_train,
                          label = train$score,
                          booster = 'gbtree',
                          nround = nround_values[i],
                          objective = "reg:squarederror")
  
  
  # Tokenizing the validation reviews and creating the iterator.
  validation_tokens <- tokenizer_function(validation$review)
  iterator_validation <- itoken(validation_tokens, 
                                ids = validation$reviewId, 
                                progressbar = FALSE)
  
  # Creating and normalizing the Document x Term matrix of the validation
  # dataset.
  document_term_matrix_validation <- create_dtm(iterator_validation, vectorizer)
  document_term_matrix_validation <- model_tf_idf$fit_transform(document_term_matrix_validation)
  
  # Precitions are generated and adjusted to our rater's scale.
  predictions <- predict(model, document_term_matrix_validation)
  predictions <- sapply(predictions, adjust_scores)
  
  # Iteration results are appended to data frame.
  results_nround_tuning <- rbind(results_nround_tuning,  data.frame(nround=nround_values[i],
                                                                    rmse=RMSE(predictions, validation$score)))
  
  # Control message.
  cat(paste('Iteration', i, 'of', length(nround_values), 'executed.\n'))
}

# With the experiment done, we save the value of Nround that produced the best
# performance (as given by RMSE). This is what we were looking for.
best_nround <- results_nround_tuning[which.min(results_nround_tuning$rmse),]$nround

# Cleaning the place after one more experiment.
rm(document_term_matrix_train, document_term_matrix_validation, model, model_tf_idf, 
   train_tokens, validation_tokens, iterator_train, iterator_validation, nround_values,
   predictions, vectorizer, i)




###############################################################################
# Step 10: Experiment 3.2 - Tuning - Max_Depth and Min_Child_Weight
#
# In this Experiment, we continue tuning the parameters with the model
# selected in Experiment 2 plus the best nround from Experiment 3.1. Here, 
# the tuning is done for max_depth and min_child_weight. 32 combinations
# will be tested.
#
# IMPORTANT NOTE: Using grid tuning from caret would have made this section
# a lot cleaner in terms of code, but then I would have had to turn the
# Document x Term Matrix into a sparse matrix and my notebook couldn't handle
# it.
###############################################################################

# This time around, we are testing two parameters. So first we create two 
# vectors with the values that will be tested.
max_depth <- seq(3,10,1)
min_child_weight <- c(1,2,4,6)

# And then we create da data frame with all of their possible combinations.
tuning_grid <- expand.grid(max_depth = max_depth, min_child_weight = min_child_weight)

# We also create the data frame that will store the results of the
# experiment. We are keeping track of the two parameters and of RMSE, of course.
results_max_depth_child_weight_tuning <- data.frame(max_depth=numeric(),
                                                    min_child_weight=numeric(),
                                                    rmse=numeric())

# The total amount of experiments will be the total number of combinations
# possible between max_depth and min_child_weight.
for (i in 1:nrow(tuning_grid)) {
  
  # The usual iteration control message.
  cat(paste('Iteration', i, 'of', nrow(tuning_grid), 'started.\n'))
  
  # The usual text vectorization preparation.
  tokenizer_function <- word_tokenizer
  train_tokens <- tokenizer_function(train$review)
  iterator_train <- itoken(train_tokens, 
                           ids = train$reviewId,
                           progressbar = FALSE)
  
  # Once more, we are only working with the vocabulary that gave us the best 
  # result.
  vocabulary <- create_vocabulary(iterator_train)
  vocabulary <- vocabulary[(vocabulary$term %in% vocabulary_best_model$term), ]
  
  # The vocabulary is used to vectorize the text and generate the Document x
  # Term matrix.
  vectorizer <- vocab_vectorizer(vocabulary)
  document_term_matrix_train <- create_dtm(iterator_train, vectorizer)
  
  # TF-IDF comes in and normalize the term counts.
  model_tf_idf <- TfIdf$new()
  document_term_matrix_train <- model_tf_idf$fit_transform(document_term_matrix_train)
  
  # Here we go again. We are setting the seed so the results can be reproduced
  # and training the model with the max_depth and min_child_weight values
  # determined by the iteration. The value of nround comes, of course, from
  # the previous experiment.
  set.seed(1, sample.kind="Rounding")
  model <- xgboost(data = document_term_matrix_train,
                   label = train$score,
                   booster = 'gbtree',
                   nround = best_nround,
                   max_depth = tuning_grid[i,]$max_depth,
                   min_child_weight = tuning_grid[i,]$min_child_weight,
                   objective = "reg:squarederror")
  
  
  # Here comes another tokenization and Document x Term matrix of the validation
  # dataset.
  validation_tokens <- tokenizer_function(validation$review)
  iterator_validation <- itoken(validation_tokens, 
                                ids = validation$reviewId, 
                                progressbar = FALSE)
  
  document_term_matrix_validation <- create_dtm(iterator_validation, vectorizer)
  document_term_matrix_validation <- model_tf_idf$fit_transform(document_term_matrix_validation)
  
  
  # Predictions are done with the trained model being applied to the validation
  # data set.
  predictions <- predict(model, document_term_matrix_validation)
  
  # Predictions are adjusted, because we are only looking for scores within the
  # 0-100 values and with 5-point increments. It's the scale of our universal
  # rater, after all.
  predictions <- sapply(predictions, adjust_scores)
  
  # Results of the current iteration are appended to the results matrix.
  results_max_depth_child_weight_tuning <- rbind(results_max_depth_child_weight_tuning, 
                                                 data.frame(max_depth=tuning_grid[i,]$max_depth,
                                                            min_child_weight=tuning_grid[i,]$min_child_weight,
                                                            rmse=RMSE(predictions, validation$score)))
  
  # Control message in case one gets anxious with these long iterations.
  cat(paste('Iteration', i, 'of', nrow(tuning_grid), 'executed.\n'))
}

# To wrap the experiment up, we get the combination of values that yielded the
# best RMSE.
best_max_depth <- results_max_depth_child_weight_tuning[which.min(results_max_depth_child_weight_tuning$rmse),]$max_depth
best_min_child_weight <- results_max_depth_child_weight_tuning[which.min(results_max_depth_child_weight_tuning$rmse),]$min_child_weight

# Making our global environment clean one more time.
rm(document_term_matrix_train, document_term_matrix_validation, model, model_tf_idf, 
   train_tokens, validation_tokens, iterator_train, iterator_validation,
   predictions, vectorizer, i, max_depth, min_child_weight, tuning_grid)




###############################################################################
# Step 11: Experiment 3.3 - Tuning - Gamma
#
# In this Experiment, we continue tuning the parameters with the model
# selected in Experiment 2 plus the best nround from Experiment 3.1, and
# the best max_depth and min_child_weight from Experiment 3.2. Here, 
# the tuning is done for gamma. 11 values will be tested.
#
# IMPORTANT NOTE: Using grid tuning from caret would have made this section
# a lot cleaner in terms of code, but then I would have had to turn the
# Document x Term Matrix into a sparse matrix and my notebook couldn't handle
# it.
###############################################################################

# Here are the values of gamma that will be tested. They are between 0 and
# 1 in increments of 0.1.
gamma_values <- seq(0,1,0.1)

# This is where we will store our experiment's results.
results_gamma_tuning <- data.frame(gamma=numeric(),
                                   rmse=numeric()) 

# This chunk of code will be repeated for every value of gamma so we can find 
# the best one for our model.
for (i in 1:length(gamma_values)) {
  
  # Just another iteration control message.
  cat(paste('Iteration', i, 'of', length(gamma_values), 'started.\n'))
  
  # Tokenizing, vectorizing, and turning the training data into a Document x
  # Term matrix.
  tokenizer_function <- word_tokenizer
  train_tokens <- tokenizer_function(train$review)
  iterator_train <- itoken(train_tokens, 
                           ids = train$reviewId,
                           progressbar = FALSE)
  vocabulary <- create_vocabulary(iterator_train)
  vocabulary <- vocabulary[(vocabulary$term %in% vocabulary_best_model$term), ]
  vectorizer <- vocab_vectorizer(vocabulary)
  document_term_matrix_train <- create_dtm(iterator_train, vectorizer)
  model_tf_idf <- TfIdf$new()
  document_term_matrix_train <- model_tf_idf$fit_transform(document_term_matrix_train)
  
  # We train the model with the parameters obtained in Steps 9-10, plus the 
  # value of gamma we are currently testing.
  set.seed(1, sample.kind="Rounding")
  model <- xgboost(data = document_term_matrix_train,
                   label = train$score,
                   booster = 'gbtree',
                   nround = best_nround,
                   max_depth = best_max_depth,
                   min_child_weight = best_min_child_weight,
                   gamma = gamma_values[i],
                   objective = "reg:squarederror")
  
  
  # Here comes another tokenization, vectorization, and transformation into
  # Document x Term matrix for the validation dataset.
  validation_tokens <- tokenizer_function(validation$review)
  iterator_validation <- itoken(validation_tokens, 
                                ids = validation$reviewId, 
                                progressbar = FALSE)
  document_term_matrix_validation <- create_dtm(iterator_validation, vectorizer)
  document_term_matrix_validation <- model_tf_idf$fit_transform(document_term_matrix_validation)
  
  # We predict, we adjust, and we store the results in our experiment's data
  # frame.
  predictions <- predict(model, document_term_matrix_validation)
  predictions <- sapply(predictions, adjust_scores)
  
  results_gamma_tuning <- rbind(results_gamma_tuning, data.frame(gamma=gamma_values[i],
                                                            rmse=RMSE(predictions, validation$score)))
  
  # Another iteration control message. Nothing special to see here.
  cat(paste('Iteration', i, 'of', length(gamma_values), 'executed.\n'))
}

# We are getting the value of gamma that gave us the best RMSE and moving on
# to the next tuning experiment.
best_gamma <- results_gamma_tuning[which.min(results_gamma_tuning$rmse),]$gamma

# But before moving forward, we clean up our mess.
rm(document_term_matrix_train, document_term_matrix_validation, model, model_tf_idf, 
   train_tokens, validation_tokens, iterator_train, iterator_validation, gamma_values,
   predictions, vectorizer, i)




###############################################################################
# Step 12: Experiment 3.4 - Tuning - Subsample and Colsample_Bytree
#
# In this Experiment, we continue tuning the parameters with the model
# selected in Experiment 2 plus the best nround from Experiment 3.1, the best 
# max_depth and min_child_weight from Experiment 3.2, and gamma from
# Experiment 3.3. Here, the tuning is done for subsample and colsample_bytree. 
# 36 combinations will be tested.
#
# IMPORTANT NOTE: Using grid tuning from caret would have made this section
# a lot cleaner in terms of code, but then I would have had to turn the
# Document x Term Matrix into a sparse matrix and my notebook couldn't handle
# it.
###############################################################################

# This is similar to Step 10, because we are tuning to parameters at the same
# time. This time it's subsample and colsample_bytree. So we create the
# vectors that will store the values we will test for them first.
subsample <- seq(0.5,1,0.1)
colsample_bytree <- seq(0.5,1,0.1)

# And then we generate all possible combinations, storing them into a data 
# frame.
tuning_grid <- expand.grid(subsample = subsample, colsample_bytree = colsample_bytree)

# Another results data frame is created.
results_subsample_colsample_tuning <- data.frame(subsample=numeric(),
                                                 colsample_bytree=numeric(),
                                                 rmse=numeric()) 

# Here we go. The number of iterations is as big as our tuning_grid.
for (i in 1:nrow(tuning_grid)) {
  
  # Another control message for the anxious.
  cat(paste('Iteration', i, 'of', nrow(tuning_grid), 'started.\n'))
  
  # Usual business here. Nothing different from before. The train reviews
  # are tokenized, vectorized, and turned into a Document x Term matrix with
  # TF-IDF applied to it.
  tokenizer_function <- word_tokenizer
  train_tokens <- tokenizer_function(train$review)
  iterator_train <- itoken(train_tokens, 
                           ids = train$reviewId,
                           progressbar = FALSE)
  vocabulary <- create_vocabulary(iterator_train)
  vocabulary <- vocabulary[(vocabulary$term %in% vocabulary_best_model$term), ]
  vectorizer <- vocab_vectorizer(vocabulary)
  document_term_matrix_train <- create_dtm(iterator_train, vectorizer)
  model_tf_idf <- TfIdf$new()
  document_term_matrix_train <- model_tf_idf$fit_transform(document_term_matrix_train)
  
  # We are setting the seed and training again. At this point, we have four
  # tuned parameters with two more being tested this time around.
  set.seed(1, sample.kind="Rounding")
  model <- xgboost(data = document_term_matrix_train,
                   label = train$score,
                   booster = 'gbtree',
                   nround = best_nround,
                   max_depth = best_max_depth,
                   min_child_weight = best_min_child_weight,
                   gamma = best_gamma,
                   subsample = tuning_grid[i,]$subsample,
                   colsample_bytree = tuning_grid[i,]$colsample_bytree,
                   objective = "reg:squarederror")

  # Here comes the validation set again to be tokenized, vectorized,
  # and have its term counts normalized.
  validation_tokens <- tokenizer_function(validation$review)
  iterator_validation <- itoken(validation_tokens, 
                                ids = validation$reviewId, 
                                progressbar = FALSE)
  document_term_matrix_validation <- create_dtm(iterator_validation, vectorizer)
  document_term_matrix_validation <- model_tf_idf$fit_transform(document_term_matrix_validation)
  
  # We predict and we adjust. Quick, simple, and already seen before.
  predictions <- predict(model, document_term_matrix_validation)
  predictions <- sapply(predictions, adjust_scores)
  
  # Appending the results of the current experient to the dataset that will
  # store them all.
  results_subsample_colsample_tuning <- rbind(results_subsample_colsample_tuning, 
                                                 data.frame(subsample=tuning_grid[i,]$subsample,
                                                            colsample_bytree=tuning_grid[i,]$colsample_bytree,
                                                            rmse=RMSE(predictions, validation$score)))
  
  # Control message to indicate iteration is done.
  cat(paste('Iteration', i, 'of', nrow(tuning_grid), 'executed.\n'))
}

# With yet another experiment done, we are getting the parameters that generated
# the best results and moving on to the next, and last, tuning step.
best_subsample <- results_subsample_colsample_tuning[which.min(results_subsample_colsample_tuning$rmse),]$subsample
best_colsample_bytree <- results_subsample_colsample_tuning[which.min(results_subsample_colsample_tuning$rmse),]$colsample_bytree

# We clean the environment one more time. It makes it all lighter and also
# also facilitates the task of checking variables and data.
rm(document_term_matrix_train, document_term_matrix_validation, model, model_tf_idf, 
   train_tokens, validation_tokens, iterator_train, iterator_validation,
   predictions, vectorizer, i, subsample, colsample_bytree, tuning_grid)




###############################################################################
# Step 13: Experiment 3.5 - Tuning - Lambda
#
# In this Experiment, we finish tuning the parameters with the model
# selected in Experiment 2 plus the best nround from Experiment 3.1, the best 
# max_depth and min_child_weight from Experiment 3.2, the best gamma from
# Experiment 3.3, and the best subsample and colsample_bytree from experiment
# 3.4. Here, the tuning is done for lambda. 
# 41 values will be tested.
#
# IMPORTANT NOTE: Using grid tuning from caret would have made this section
# a lot cleaner in terms of code, but then I would have had to turn the
# Document x Term Matrix into a sparse matrix and my notebook couldn't handle
# it.
###############################################################################

# Our beloved regularization parameter. Here there was a choice between alpha
# (L1 regularization) and lambda (L2 regularization). The latter was chosen for
# being the most conservative of the two (it doesn't take out features
# completely like Lasso does, it just reduces their weight).
lambda_values <- seq(0,20,0.5)

# The always present results data frame.
results_lambda_tuning <- data.frame(lambda=numeric(),
                                    rmse=numeric()) 

# There are quite a few values of lambda to be tested, and this will guarantee
# all of them are evaluated. The usual.
for (i in 1:length(lambda_values)) {
  
  # And the usual control message.
  cat(paste('Iteration', i, 'of', length(lambda_values), 'started.\n'))
  
  # The train dataset is tokenized, vectorized, and turned into a Document
  # x Term matrix.
  tokenizer_function <- word_tokenizer
  train_tokens <- tokenizer_function(train$review)
  iterator_train <- itoken(train_tokens, 
                           ids = train$reviewId,
                           progressbar = FALSE)
  vocabulary <- create_vocabulary(iterator_train)
  vocabulary <- vocabulary[(vocabulary$term %in% vocabulary_best_model$term), ]
  vectorizer <- vocab_vectorizer(vocabulary)
  document_term_matrix_train <- create_dtm(iterator_train, vectorizer)
  model_tf_idf <- TfIdf$new()
  document_term_matrix_train <- model_tf_idf$fit_transform(document_term_matrix_train)
  
  # Almost all of our targeted parameters are set, with the exception of one.
  # Here we put them all in place, as usual, and alternate the value of the one 
  # we are tuning.
  set.seed(1, sample.kind="Rounding")
  model <- xgboost(data = document_term_matrix_train,
                   label = train$score,
                   booster = 'gbtree',
                   nround = best_nround,
                   max_depth = best_max_depth,
                   min_child_weight = best_min_child_weight,
                   gamma = best_gamma,
                   subsample = best_subsample,
                   colsample_bytree = best_colsample_bytree,
                   lambda = lambda_values[i],
                   objective = "reg:squarederror")
  
  # The validation transformation into a Document x Term matrix is briefer
  # because we already have the vocabulary and vectorizer ready.
  validation_tokens <- tokenizer_function(validation$review)
  iterator_validation <- itoken(validation_tokens, 
                                ids = validation$reviewId, 
                                progressbar = FALSE)
  document_term_matrix_validation <- create_dtm(iterator_validation, vectorizer)
  document_term_matrix_validation <- model_tf_idf$fit_transform(document_term_matrix_validation)
  
  # We predict and we adjust our predicted scores to our scale.
  predictions <- predict(model, document_term_matrix_validation)
  predictions <- sapply(predictions, adjust_scores)
  
  # Results of the experiment are recorded into the result data frame of the
  # experiment to keep everything neat and recorded.
  results_lambda_tuning <- rbind(results_lambda_tuning, data.frame(lambda=lambda_values[i],
                                                                   rmse=RMSE(predictions, validation$score)))
  
  # Control message. The last.
  cat(paste('Iteration', i, 'of', length(lambda_values), 'executed.\n'))
}

# We get the best lambda, and now we are ready for the final results. We have
# tuned everything we set out to tune.
best_lambda <- results_lambda_tuning[which.min(results_lambda_tuning$rmse),]$lambda

# We clean the environment to get ready for the final step.
rm(document_term_matrix_train, document_term_matrix_validation, model, model_tf_idf, 
   train_tokens, validation_tokens, iterator_train, iterator_validation, lambda_values,
   predictions, vectorizer, i)




###############################################################################
# Step 14: Final Results
#
# In this final part of the code, the tuned model is trained on the full
# training + validation dataset and it is applied on the test set so we 
# can obtain the final results.
###############################################################################

# The validation dataset has served its purpose and now we will use it not to
# check our results (that will be the job of the test data set), but to train
# our algorithm alongside the training records. In other words, the original
# training dataset that we created and split even further before the experiments
# were executed is now made whole again so we train with all the data we have at
# our disposal (which excludes the test set, of course).
full_training_ds <- rbind(train, validation)

# We tokenize, vectorize, and turn our full training dataset into a Document x
# Term matrix.
tokenizer_function <- word_tokenizer
train_tokens <- tokenizer_function(full_training_ds$review)
iterator_train <- itoken(train_tokens, 
                         ids = full_training_ds$reviewId,
                         progressbar = FALSE)

# Note that, like we have been doing since Section 8, we are using the 
# vocabulary we got to via Feature Selection; i.e, the terms that generated
# the most precise model.
vocabulary <- create_vocabulary(iterator_train)
vocabulary <- vocabulary[(vocabulary$term %in% vocabulary_best_model$term), ]
vectorizer <- vocab_vectorizer(vocabulary)
document_term_matrix_train <- create_dtm(iterator_train, vectorizer)
model_tf_idf <- TfIdf$new()
document_term_matrix_train <- model_tf_idf$fit_transform(document_term_matrix_train)

# The model is trained with all the tuned parameters. That is, the ones selected
# as the best in Sections 9-13. So this is the culmination of all that has been
# done: the best model, with the best vocabulary, and the best parameters.
set.seed(1, sample.kind="Rounding")
final_model <- xgboost(data = document_term_matrix_train,
                       label = full_training_ds$score,
                       booster = 'gbtree',
                       nround = best_nround,
                       max_depth = best_max_depth,
                       min_child_weight = best_min_child_weight,
                       gamma = best_gamma,
                       subsample = best_subsample,
                       colsample_bytree = best_colsample_bytree,
                       lambda = best_lambda,
                       objective = "reg:squarederror")

# Now, the test dataset enters the scene for the first time since the split.
# We tokenize it and vectorize it using the vocabulary we chose to consider
# (the one reached through Feature Selection). And then we produce a
# Document x Term matrix.

test_tokens <- tokenizer_function(test$review)
iterator_teste <- itoken(test_tokens, 
                        ids = test$reviewId, 
                        progressbar = FALSE)
document_term_matrix_test <- create_dtm(iterator_teste, vectorizer)

# We normalize its vocabulary counts via the TF-IDF that was fit with the
# training data.
document_term_matrix_test <- model_tf_idf$fit_transform(document_term_matrix_test)

# And, finally, we process it with our final model, getting the predictions
# that will yield the final result of this work.
predictions <- predict(final_model, document_term_matrix_test)

# The predictions are adjusted before they are verified, because our targeted
# Universal Rater only works in increments of 5 points. That is our intended
# global scoring scale.
adjusted_predictions <- sapply(predictions, adjust_scores)

# We get the final RMSE and tuck it into a variable. This is the ultimate
# result.
final_rmse <- RMSE(adjusted_predictions, test$score)

# For reporting purposes, though, one last task is done. First, we get the test
# set data frame and append some columns to it: the adjusted prediction, and
# the non-adjusted prediction.
final_results <- cbind(test, adjusted_predictions, non_adjusted_predictions = predictions)

# Moreover, we create a column called "difference", which will tell us how far
# our predictions were from the actual score.
final_results <- final_results %>% mutate(difference = abs(score - adjusted_predictions))

# Finally, we do some joining and mutations. For reporting and result 
# visualization purposes: we create a column transformed_review to store
# the review in its state after all our text processing (that is, in the format
# it was used to make the prediction); we rescue the original review, in its
# untouched state, from the original_reviews data frame, and we do some cleaning
# up by renaming columns and selecting just the ones we want after the join.
final_results <- final_results %>% 
  mutate(transformed_review = review) %>%
  select(reviewId, transformed_review, score, non_adjusted_predictions, adjusted_predictions, difference) %>%
  inner_join(original_reviews, by="reviewId") %>% 
  mutate(original_review = review, score=score.x) %>%
  select(reviewId, original_review, transformed_review, score, non_adjusted_predictions, adjusted_predictions, difference)

# And we clean the environment one last time so all that we have left is the
# objects that are important for the observation of the final results.
rm(document_term_matrix_test, document_term_matrix_train, full_training_ds, model_tf_idf, test_tokens,
   train_tokens, vocabulary, adjusted_predictions, iterator_teste, iterator_train, predictions,
   tokenizer_function, vectorizer, adjust_scores, obtain_importance_terms, remove_game_name_from_review_text,
   remove_words_from_text)
