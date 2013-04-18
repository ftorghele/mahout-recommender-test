package org.movlib;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.List;
import java.io.IOException;

import org.apache.commons.cli2.OptionException; 
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.recommender.CachingRecommender;
import org.apache.mahout.cf.taste.impl.recommender.slopeone.SlopeOneRecommender;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;

public class UserBasedRecommend {
    
    public static void main(String... args) throws FileNotFoundException, TasteException, IOException, OptionException {
        
        // create data source (model) - from the csv file            
        File ratingsFile = new File("datasets/bool-recommend-dummy.csv");                        
        DataModel model = new FileDataModel(ratingsFile);
        
        // create a simple recommender on our data
        CachingRecommender cachingRecommender = new CachingRecommender(new SlopeOneRecommender(model));

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