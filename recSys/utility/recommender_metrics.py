##-----------------------------------------------------------------------------------##
#
# Source: Building Recommender Systems with Machine Learning and AI, Sundog Education
#
##----------------------------------------------------------------------------------##

### different metrics for evaluating recsys 
import itertools

from surprise import accuracy
from collections import defaultdict

class RecommenderMetrics:

    def mae(predictions):
        """
        Returns
            mean absolute error - smaller is better
        """
        return accuracy.mae(predictions, verbose=False)

    def rmse(predictions):
        """
        Returns
            root mean squared error - smaller is better
        """
        return accuracy.rmse(predictions, verbose=False)
    
    def fcp(predictions):
        """
        Reference: https://www.ijcai.org/Proceedings/13/Papers/449.pdf
        Returns
            fraction of concordant pairs - larger is better
        """
        return accuracy.fcp(predictions, verbose=False)

    def get_top_n(predictions, n=10, minimum_rating=4.0):
        """
        Returns
            a dictionary with top_n (default is 10) estimated ratings / each user
        """
        top_n = defaultdict(list)

        for user_id, movie_id, actual_rating, estimated_rating, _ in predictions:
            if (estimated_rating >= minimum_rating):
                top_n[int(user_id)].append((int(movie_id), estimated_rating))

        for user_id, ratings in top_n.items():
            ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[int(user_id)] = ratings[:n]

        return top_n

    def hit_rate(top_n_predicted, left_out_predictions):
        """
        Returns
            hit rate - larger is better
        """
        hits = 0
        total = 0

        # For each left-out rating
        for left_out in left_out_predictions:
            user_id = left_out[0]
            left_out_movie_id = left_out[1]
            # Is it in the predicted top 10 for this user?
            hit = False
            for movie_id, predicted_rating in top_n_predicted[int(user_id)]:
                if (int(left_out_movie_id) == int(movie_id)):
                    hit = True
                    break
            if (hit) :
                hits += 1

            total += 1

        # Compute overall precision
        return hits/total

    def cummulative_hit_rate(top_n_predicted, left_out_predictions, rating_cutoff=0):
        hits = 0
        total = 0

        # For each left-out rating
        for user_id, left_out_movie_id, actual_rating, estimated_rating, _ in left_out_predictions:
            # Only look at ability to recommend things the users actually liked...
            if (actual_rating >= rating_cutoff):
                # Is it in the predicted top 10 for this user?
                hit = False
                for movie_id, predicted_rating in top_n_predicted[int(user_id)]:
                    if (int(left_out_movie_id) == movie_id):
                        hit = True
                        break
                if (hit) :
                    hits += 1

                total += 1

        # Compute overall precision
        return hits/total

    def rating_hit_rate(top_n_predicted, left_out_predictions):
        hits = defaultdict(float)
        total = defaultdict(float)

        # For each left-out rating
        for user_id, left_out_movie_id, actual_rating, estimated_rating, _ in left_out_predictions:
            # Is it in the predicted top N for this user?
            hit = False
            for movie_id, predicted_rating in top_n_predicted[int(user_id)]:
                if (int(left_out_movie_id) == movie_id):
                    hit = True
                    break
            if (hit) :
                hits[actual_rating] += 1

            total[actual_rating] += 1

        # Compute overall precision
        for rating in sorted(hits.keys()):
            print (rating, hits[rating] / total[rating])

    def average_reciprocal_hit_rank(top_n_predicted, left_out_predictions):
        summation = 0
        total = 0
        # For each left-out rating
        for user_id, left_out_movie_id, actual_rating, estimated_rating, _ in left_out_predictions:
            # Is it in the predicted top N for this user?
            hit_rank = 0
            rank = 0
            for movie_id, predicted_rating in top_n_predicted[int(user_id)]:
                rank = rank + 1
                if (int(left_out_movie_id) == movie_id):
                    hit_rank = rank
                    break
            if (hit_rank > 0) :
                summation += 1.0 / hit_rank

            total += 1

        return summation / total

    # What percentage of users have at least one "good" recommendation
    def user_coverage(top_n_predicted, num_users, rating_threshold=0):
        hits = 0
        for user_id in top_n_predicted.keys():
            hit = False
            for movie_id, predicted_rating in top_n_predicted[int(user_id)]:
                if (predicted_rating >= rating_threshold):
                    hit = True
                    break
            if (hit):
                hits += 1

        return hits / num_users

    def diversity(top_n_predicted, sims_algo):
        n = 0
        total = 0
        sims_matrix = sims_algo.compute_similarities()
        for user_id in top_n_predicted.keys():
            pairs = itertools.combinations(top_n_predicted[user_id], 2)
            for pair in pairs:
                movie_1 = pair[0][0]
                movie_2 = pair[1][0]
                inner_id_1 = sims_algo.trainset.to_inner_iid(movie_1)
                inner_id_2 = sims_algo.trainset.to_inner_iid(movie_2)
                similarity = sims_matrix[inner_id_1][inner_id_2]
                total += similarity
                n += 1

        if n > 0:
            avg_sim = total / n
        else:
            avg_sim = 0.0
        return (1- avg_sim)

    def novelty(top_n_predicted, rankings):
        n = 0
        total = 0
        for user_id in top_n_predicted.keys():
            for rating in top_n_predicted[user_id]:
                movie_id = rating[0]
                rank = rankings[movie_id]
                total += rank
                n += 1
        if n > 0:
            return total / n
        return 0.0