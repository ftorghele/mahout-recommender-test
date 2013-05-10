package org.movlib;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Random;

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
import org.apache.mahout.common.RandomUtils;

import com.google.common.collect.Lists;

/**
 * if out of memory: export MAVEN_OPTS="-Xmx1024m"
 */

public class CrossValidationEvaluator {
	
	public static void main(String... args) throws IOException, TasteException,
			OptionException {

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

		// prepare Datasets
		int setCount = 10;

		DataModel[] tenFoldDataset = prepareTenFold(allRecommendations, setCount);
		DataModel[] bootstrappedDataset = prepareBootstrap(allRecommendations, setCount);
		//DataModel[] jackknifedDataset = prepareJackknife(allRecommendations, setCount);
		
		// 10-fold cross validation
		double tenFoldRMSE = 0.0;
		System.out.println("\n10-fold cross validation:");
		for (int i = 0; i < setCount; i++) {
			DataModel trainingModel = prepareTrainingSet(tenFoldDataset, i);
			System.out.print(".");
			DataModel testModel = tenFoldDataset[i];
			CustomEvaluator evaluator = new CustomEvaluator();
			tenFoldRMSE += evaluator.evaluate(builder, null, trainingModel,	testModel);
		}
		System.out.println("\nRMSE: " + tenFoldRMSE/setCount);
		
		// Bootstrapped 10-fold cross validation
		double bootstrappedRMSE = 0.0;
		System.out.println("\nBootstrapped 10-fold cross validation:");
		for (int i = 0; i < setCount; i++) {
			DataModel trainingModel = prepareTrainingSet(bootstrappedDataset, i);
			System.out.print(".");
			DataModel testModel = bootstrappedDataset[i];
			CustomEvaluator evaluator = new CustomEvaluator();
			bootstrappedRMSE += evaluator.evaluate(builder, null, trainingModel,	testModel);
		}
		System.out.println("\nRMSE: " + bootstrappedRMSE/setCount);
		
		// Jackknifed 10-fold cross validation
//		double jackknifedRMSE = 0.0;
//		System.out.println("Jackknifed 10-fold cross validation:");
//		for (int i = 0; i < setCount; i++) {
//			DataModel trainingModel = prepareTrainingSet(jackknifedDataset, i);
//			System.out.print(".");
//			DataModel testModel = jackknifedDataset[i];
//			CustomEvaluator evaluator = new CustomEvaluator();
//			jackknifedRMSE += evaluator.evaluate(builder, null, trainingModel,	testModel);
//		}
//		System.out.println("\nRMSE: " + jackknifedRMSE/setCount);
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
		
		Random random = RandomUtils.getRandom();

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
			int numPrefs = userPrefs.length();
			for (int i = 0; i < numPrefs; i++) {

				Preference newPref = new GenericPreference(userID,
						userPrefs.getItemID(random.nextInt( numPrefs - 1 )), 
						userPrefs.getValue(i));
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
	
//	@SuppressWarnings("unchecked")
//	private static DataModel[] prepareJackknife(DataModel dataModel, int setCount) throws TasteException {
//		
//		int numUsers = dataModel.getNumUsers();
//
//		@SuppressWarnings("rawtypes")
//		FastByIDMap[] resultPrefs = new FastByIDMap[setCount];
//
//		for (int i = 0; i < setCount; i++) {
//			resultPrefs[i] = new FastByIDMap<PreferenceArray>(numUsers + 1);
//		}
//
//		// TODO: implement jackknife
//		
//		DataModel[] result = new DataModel[setCount];
//
//		for (int i = 0; i < setCount; i++) {
//			result[i] = new GenericDataModel(resultPrefs[i]);
//		}
//
//		return result;
//	}

}