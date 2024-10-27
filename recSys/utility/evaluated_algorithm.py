##-----------------------------------------------------------------------------------##
#
# Source: adapted for Kaggle from 
#         Building Recommender Systems with Machine Learning and AI, Sundog Education
#
##----------------------------------------------------------------------------------##

### evaluate performace of algorithm by calculating different metrics

from utility.recommender_metrics import RecommenderMetrics
from utility.evaluation_data import EvaluationData

class EvaluatedAlgorithm:
    
    def __init__(self, algorithm, name):
        self.algorithm = algorithm
        self.name = name
        
    def evaluate(self, evaluation_data, do_top_n, n=10, verbose=True):
        metrics = {}
        # Compute accuracy
        if (verbose):
            print("Evaluating accuracy...")
        self.algorithm.fit(evaluation_data.get_train_set())
        predictions = self.algorithm.test(evaluation_data.get_test_set())
        metrics["RMSE"] = RecommenderMetrics.rmse(predictions)
        metrics["MAE"] = RecommenderMetrics.mae(predictions)
        metrics["FCP"] = RecommenderMetrics.fcp(predictions)
        
        if (do_top_n):
            # Evaluate top-10 with Leave One Out testing
            if (verbose):
                print("Evaluating top-N with leave-one-out...")
            self.algorithm.fit(evaluation_data.get_loocv_train_set())
            left_out_predictions = self.algorithm.test(evaluation_data.get_loocv_test_set())        
            # Build predictions for all ratings not in the training set
            all_predictions = self.algorithm.test(evaluation_data.get_loocv_anti_test_set())
            # Compute top 10 recs for each user
            top_n_predicted = RecommenderMetrics.get_top_n(all_predictions, n)
            if (verbose):
                print("Computing hit-rate and rank metrics...")
            # See how often we recommended a movie the user actually rated
            metrics["HR"] = RecommenderMetrics.hit_rate(top_n_predicted, left_out_predictions)   
            # See how often we recommended a movie the user actually liked
            metrics["cHR"] = RecommenderMetrics.cummulative_hit_rate(top_n_predicted, left_out_predictions)
            # Compute ARHR
            metrics["ARHR"] = RecommenderMetrics.average_reciprocal_hit_rank(top_n_predicted, left_out_predictions)
        
            #Evaluate properties of recommendations on full training set
            if (verbose):
                print("Computing recommendations with full data set...")
            self.algorithm.fit(evaluation_data.get_full_train_set())
            all_predictions = self.algorithm.test(evaluation_data.get_full_anti_test_set())
            top_n_predicted = RecommenderMetrics.get_top_n(all_predictions, n)
            if (verbose):
                print("Analyzing coverage, diversity, and novelty...")
            # Print user coverage with a minimum predicted rating of 4.0:
            metrics["Coverage"] = RecommenderMetrics.user_coverage(top_n_predicted, 
                                                                   evaluation_data.get_full_train_set().n_users, 
                                                                   rating_threshold=4.0)
            # Measure diversity of recommendations:
            metrics["Diversity"] = RecommenderMetrics.diversity(top_n_predicted, evaluation_data.get_similarities())
            
            # Measure novelty (average popularity rank of recommendations):
            metrics["Novelty"] = RecommenderMetrics.novelty(top_n_predicted, 
                                                            evaluation_data.get_popularity_rankings())
        
        if (verbose):
            print("Analysis complete.")
    
        return metrics
    
    def get_name(self):
        return self.name
    
    def get_algorithm(self):
        return self.algorithm