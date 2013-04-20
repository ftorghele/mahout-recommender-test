package org.movlib;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.List;
import java.io.IOException;

import org.apache.commons.cli2.OptionException; 
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.CachingRecommender;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;

public class UserBasedRecommend {
    
    public static void main(String... args) throws FileNotFoundException, TasteException, IOException, OptionException {
        
        // create data source (model) - from the csv file            
        File ratingsFile = new File("datasets/movielens/ratings.csv");                        
        DataModel model = new FileDataModel(ratingsFile);
        
        // We'll use the PearsonCorrelationSimilarity implementation of UserSimilarity 
        // as our user correlation algorithm, and add an optional preference inference algorithm:
        UserSimilarity userSimilarity = new PearsonCorrelationSimilarity(model);
        // Optional:
        //userSimilarity.setPreferenceInferrer(new AveragingPreferenceInferrer());

        // Now we create a UserNeighborhood algorithm. Here we use nearest-3:
        UserNeighborhood neighborhood = new NearestNUserNeighborhood(3, userSimilarity, model);
        // Now we can create our Recommender, and add a caching decorator:
        Recommender recommender = new GenericUserBasedRecommender(model, neighborhood, userSimilarity);
        Recommender cachingRecommender = new CachingRecommender(recommender);

        // for all users
        for (LongPrimitiveIterator it = model.getUserIDs(); it.hasNext();){
            long userId = it.nextLong();
            
            // get the recommendations for the user
            List<RecommendedItem> recommendations = cachingRecommender.recommend(userId, 10);
            
            // if empty write something
            if (recommendations.size() == 0){
                System.out.print("User ");
                System.out.print(userId);
                System.out.println(": no recommendations");
            }
                            
            // print the list of recommendations for each 
            for (RecommendedItem recommendedItem : recommendations) {
                System.out.print("User ");
                System.out.print(userId);
                System.out.print(": ");
                System.out.println(recommendedItem);
            }
        }        
    }
}