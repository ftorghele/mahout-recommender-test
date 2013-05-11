package org.movlib;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.apache.commons.cli2.OptionException;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.eval.CustomEvaluator;
import org.apache.mahout.cf.taste.impl.model.GenericDataModel;
import org.apache.mahout.cf.taste.impl.model.GenericPreference;
import org.apache.mahout.cf.taste.impl.model.GenericUserPreferenceArray;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.CachingRecommender;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.CustomSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;
import com.google.common.collect.Lists;

/**
 * if out of memory: export MAVEN_OPTS="-Xmx1024m"
 * mvn exec:java -Dexec.mainClass="org.movlib.CrossValidationEvaluator"
 */

public class CrossValidationEvaluator {
	
	public static void main(String... args) throws IOException, TasteException,
			OptionException {
		
		int setCount = 10;

		// get all recommendations
		DataModel allRecommendations = new FileDataModel(new File(
				"datasets/ml-100k/ratings.csv"));
		
		// build Custom Recommender
		RecommenderBuilder builder = new RecommenderBuilder() {
			public Recommender buildRecommender(DataModel model)
					throws TasteException {
				UserSimilarity userSimilarity = new CustomSimilarity(model);
				UserNeighborhood neighborhood = new NearestNUserNeighborhood(3,
						userSimilarity, model);
				Recommender recommender = new GenericUserBasedRecommender(
						model, neighborhood, userSimilarity);
				return new CachingRecommender(recommender);
			}
		};
		
		// 10-fold cross validation
		double tenFoldRMSE = 0.0;
		System.out.println("\n" + setCount + "-fold cross validation:");
		
		DataModel[] tenFoldDataset = prepareTenFold(allRecommendations, setCount);
		
		for (int i = 0; i < setCount; i++) {
			DataModel trainingModel = prepareTrainingSet(tenFoldDataset, i);
			DataModel testModel = tenFoldDataset[i];
			CustomEvaluator evaluator = new CustomEvaluator();
			double RMSE = evaluator.evaluate(builder, null, trainingModel, testModel); 
			tenFoldRMSE += RMSE;
			System.out.println((i+1) + "/"+ setCount + " RSME: " + RMSE);
		}
		System.out.println("\nRMSE avg: " + tenFoldRMSE/setCount);
		
		// Bootstrapped 10-fold cross validation
		double bootstrappedRMSE = 0.0;
		System.out.println("\nBootstrapped " + setCount + "-fold cross validation:");
		for (int i = 0; i < setCount; i++) {
			
			DataModel[] bootstrappedDataset = prepareBootstrap(allRecommendations, setCount);
			
			DataModel trainingModel = prepareTrainingSet(bootstrappedDataset, i);
			DataModel testModel = bootstrappedDataset[i];
			CustomEvaluator evaluator = new CustomEvaluator();
			double RMSE = evaluator.evaluate(builder, null, trainingModel, testModel); 
			bootstrappedRMSE += RMSE;
			System.out.println((i+1) + "/"+ setCount + " RSME: " + RMSE);
		}
		System.out.println("\nRMSE avg: " + bootstrappedRMSE/setCount);
		
		// Jackknifed 10-fold cross validation
//		double jackknifedRMSE = 0.0;
//		System.out.println("Jackknifed " + setCount + "-fold-fold cross validation:");
//		for (int i = 0; i < setCount; i++) {
//		
//		    DataModel[] jackknifedDataset = prepareJackknife(allRecommendations, setCount);
//		
//			DataModel trainingModel = prepareTrainingSet(jackknifedDataset, i);
//			DataModel testModel = jackknifedDataset[i];
//			CustomEvaluator evaluator = new CustomEvaluator();
//			double RMSE = evaluator.evaluate(builder, null, trainingModel, testModel); 
//			jackknifedRMSE += RMSE;
//			System.out.println((i+1) + "/"+ setCount + " RSME: " + RMSE);
//		}
//		System.out.println("\nRMSE avg: " + jackknifedRMSE/setCount);
	}
	
	private static DataModel prepareTrainingSet(DataModel[] dataModels,
			int skipSetNumber) throws TasteException {
		
		int numUsers = dataModels[0].getNumUsers();

		FastByIDMap<PreferenceArray> resultPrefs = new FastByIDMap<PreferenceArray>(numUsers + 1);

		LongPrimitiveIterator it = dataModels[0].getUserIDs();
		while (it.hasNext()) {
			long userID = it.nextLong();

			List<Preference> userPrefs = Lists.newArrayListWithCapacity(3);

			for (int i = 0; i < dataModels.length; i++) {
				if (i != skipSetNumber) {
					PreferenceArray prefs = dataModels[i]
							.getPreferencesFromUser(userID);
					for (int j = 0; j < prefs.length(); j++) {
						Preference newPref = new GenericPreference(userID,
								prefs.getItemID(j), prefs.getValue(j));
						userPrefs.add(newPref);
					}
				}
			}
			resultPrefs.put(userID, new GenericUserPreferenceArray(userPrefs));
		}

		return new GenericDataModel(resultPrefs);
	}

	@SuppressWarnings("unchecked")
	private static DataModel[] prepareTenFold(DataModel dataModel, int setCount) throws TasteException {
		
		int numUsers = dataModel.getNumUsers();

		@SuppressWarnings("rawtypes")
		FastByIDMap[] resultPrefs = new FastByIDMap[setCount];

		for (int i = 0; i < setCount; i++) {
			resultPrefs[i] = new FastByIDMap<PreferenceArray>(numUsers + 1);
		}

		int prefCount = 0;

		LongPrimitiveIterator it = dataModel.getUserIDs();
		while (it.hasNext()) {
			long userID = it.nextLong();

			List<Preference>[] userPreferances = new List[setCount];

			for (int i = 0; i < setCount; i++) {
				userPreferances[i] = Lists.newArrayListWithCapacity(3);
			}

			PreferenceArray userPrefs = dataModel
					.getPreferencesFromUser(userID);

			for (int i = 0; i < userPrefs.length(); i++) {

				Preference newPref = new GenericPreference(userID,
						userPrefs.getItemID(i), userPrefs.getValue(i));

				userPreferances[prefCount % setCount].add(newPref);
				prefCount++;
			}

			for (int i = 0; i < setCount; i++) {
				resultPrefs[i].put(userID, new GenericUserPreferenceArray(
						userPreferances[i]));
			}
		}

		DataModel[] result = new DataModel[setCount];

		for (int i = 0; i < setCount; i++) {
			result[i] = new GenericDataModel(resultPrefs[i]);
		}

		return result;
	}
	
	@SuppressWarnings("unchecked")
	private static DataModel[] prepareBootstrap(DataModel dataModel, int setCount) throws TasteException {
		
		int numUsers = dataModel.getNumUsers();

		@SuppressWarnings("rawtypes")
		FastByIDMap[] resultPrefs = new FastByIDMap[setCount];

		for (int i = 0; i < setCount; i++) {
			resultPrefs[i] = new FastByIDMap<PreferenceArray>(numUsers + 1);
		}
		
		int prefCount = 0;

		LongPrimitiveIterator it = dataModel.getUserIDs();
		while (it.hasNext()) {
			long userID = it.nextLong();

			List<Preference>[] userPreferances = new List[setCount];

			for (int i = 0; i < setCount; i++) {
				userPreferances[i] = Lists.newArrayListWithCapacity(3);
			}

			PreferenceArray userPrefs = dataModel.getPreferencesFromUser(userID);
			
			int numPrefs = userPrefs.length();
			ArrayList<Integer>randomIndices = new ArrayList<Integer>(numPrefs);

	        for (int i = 0; i < numPrefs; i++) {                
	        	randomIndices.add(i);
	        }

	        Collections.shuffle(randomIndices);
			
			for (int i = 0; i < (int)(numPrefs/2); i++) {

				Preference newPref = new GenericPreference(userID,
						userPrefs.getItemID(randomIndices.get(i)), 
						userPrefs.getValue(randomIndices.get(i)));
				userPreferances[prefCount % setCount].add(newPref);
				prefCount++;
			}

			for (int i = 0; i < setCount; i++) {
				resultPrefs[i].put(userID, new GenericUserPreferenceArray(
						userPreferances[i]));
			}
		}
				
		DataModel[] result = new DataModel[setCount];

		for (int i = 0; i < setCount; i++) {
			result[i] = new GenericDataModel(resultPrefs[i]);
		}

		return result;
	}
	
//	private static DataModel prepareJackknife(DataModel dataModel) throws TasteException {
//		
//		int numUsers = dataModel.getNumUsers();
//
//		FastByIDMap resultPrefs = new FastByIDMap<PreferenceArray>(numUsers + 1);
//
//		// TODO: implement jackknife
//		
//		return new GenericDataModel(resultPrefs);;
//	}

}