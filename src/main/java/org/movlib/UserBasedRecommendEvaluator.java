package org.movlib;

import java.io.File;
import java.io.IOException;
 
import org.apache.commons.cli2.OptionException;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.RMSRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.CachingRecommender;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.AveragingPreferenceInferrer;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;
 
public final class UserBasedRecommendEvaluator{
	public static void main(String... args) throws IOException, TasteException, OptionException {
 
		RecommenderBuilder builder = new RecommenderBuilder() {
			public Recommender buildRecommender(DataModel model) throws TasteException{        
		        // We'll use the PearsonCorrelationSimilarity implementation of UserSimilarity 
		        // as our user correlation algorithm, and add an optional preference inference algorithm:
		        UserSimilarity userSimilarity = new PearsonCorrelationSimilarity(model);
		        // Optional:
		        userSimilarity.setPreferenceInferrer(new AveragingPreferenceInferrer(model));

		        // Now we create a UserNeighborhood algorithm. Here we use nearest-3:
		        UserNeighborhood neighborhood = new NearestNUserNeighborhood(3, userSimilarity, model);
		        // Now we can create our Recommender, and add a caching decorator:
		        Recommender recommender = new GenericUserBasedRecommender(model, neighborhood, userSimilarity);
		        return new CachingRecommender(recommender);
		        
		        //return new CachingRecommender(new SlopeOneRecommender(model));
			}
		};
 
		RecommenderEvaluator evaluator = new RMSRecommenderEvaluator();
		DataModel model = new FileDataModel(new File("datasets/movielens/ml-100k/ratings.csv"));
		double score = evaluator.evaluate(builder,
				null,
				model,
				0.8,
				1);
 
		System.out.println(score);
	}
}