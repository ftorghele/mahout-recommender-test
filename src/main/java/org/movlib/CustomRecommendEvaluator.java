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
import org.apache.mahout.cf.taste.impl.similarity.CustomSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;
 
public final class CustomRecommendEvaluator{
	public static void main(String... args) throws IOException, TasteException, OptionException {
 
		RecommenderBuilder builder = new RecommenderBuilder() {
			public Recommender buildRecommender(DataModel model) throws TasteException{        
		        UserSimilarity userSimilarity = new CustomSimilarity(model);
		        UserNeighborhood neighborhood = new NearestNUserNeighborhood(3, userSimilarity, model);
		        Recommender recommender = new GenericUserBasedRecommender(model, neighborhood, userSimilarity);
		        return new CachingRecommender(recommender);
			}
		};
		
		RecommenderEvaluator evaluator = new RMSRecommenderEvaluator();
		DataModel model = new FileDataModel(new File("datasets/ml-100k/ratings.csv"));
		double score = evaluator.evaluate(builder,
				null,
				model,
				0.8,
				1);
 
		System.out.println(score);
	}
}