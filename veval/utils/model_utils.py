def trim_predictions_to_max_token_length(tokenizer, prediction, max_token_length=128):
   """Trims prediction output to `max_token_length` tokens"""
   tokenized_prediction = tokenizer.encode(prediction)
   trimmed_tokenized_prediction = tokenized_prediction[1: max_token_length+1]
   trimmed_prediction = tokenizer.decode(trimmed_tokenized_prediction)
   return trimmed_prediction