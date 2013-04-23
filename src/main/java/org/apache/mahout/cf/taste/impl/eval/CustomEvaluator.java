package org.apache.mahout.cf.taste.impl.eval;

import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.atomic.AtomicInteger;

import com.google.common.collect.Lists;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.DataModelBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverageAndStdDev;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.impl.common.RunningAverageAndStdDev;
import org.apache.mahout.cf.taste.impl.model.GenericDataModel;
import org.apache.mahout.cf.taste.impl.model.GenericPreference;
import org.apache.mahout.cf.taste.impl.model.GenericUserPreferenceArray;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.common.RandomUtils;

public final class CustomEvaluator extends AbstractDifferenceRecommenderEvaluator {
  
  private final Random random;
	
  private RunningAverage average;
  
  public CustomEvaluator() {
    random = RandomUtils.getRandom();
  }
  
  @Override
  public double evaluate(RecommenderBuilder recommenderBuilder,
                         DataModelBuilder dataModelBuilder,
                         DataModel dataModel,
                         double trainingPercentage,
                         double evaluationPercentage) throws TasteException {
    
    int numUsers = dataModel.getNumUsers();
    FastByIDMap<PreferenceArray> trainingPrefs = new FastByIDMap<PreferenceArray>(
        1 + (int) (evaluationPercentage * numUsers));
    FastByIDMap<PreferenceArray> testPrefs = new FastByIDMap<PreferenceArray>(
        1 + (int) (evaluationPercentage * numUsers));
    
    LongPrimitiveIterator it = dataModel.getUserIDs();
    while (it.hasNext()) {
      long userID = it.nextLong();
      if (random.nextDouble() < evaluationPercentage) {
        splitOneUsersPrefs(trainingPercentage, trainingPrefs, testPrefs, userID, dataModel);
      }
    }
    
    DataModel trainingModel = dataModelBuilder == null ? new GenericDataModel(trainingPrefs)
        : dataModelBuilder.buildDataModel(trainingPrefs);
    
    Recommender recommender = recommenderBuilder.buildRecommender(trainingModel);
    
    double result = getEvaluation(testPrefs, recommender);
    return result;
  }
  
  @Override
  protected void reset() {
    average = new FullRunningAverage();
  }
  
  @Override
  protected void processOneEstimate(float estimatedPreference, Preference realPref) {
    double diff = realPref.getValue() - estimatedPreference;
    average.addDatum(diff * diff);
  }
  
  @Override
  protected double computeFinalEvaluation() {
    return Math.sqrt(average.getAverage());
  }
  
  @Override
  public String toString() {
    return "CustomEvaluator";
  }
  
	private void splitOneUsersPrefs(double trainingPercentage,
			FastByIDMap<PreferenceArray> trainingPrefs,
			FastByIDMap<PreferenceArray> testPrefs, long userID,
			DataModel dataModel) throws TasteException {
		List<Preference> oneUserTrainingPrefs = null;
		List<Preference> oneUserTestPrefs = null;
		PreferenceArray prefs = dataModel.getPreferencesFromUser(userID);
		int size = prefs.length();
		for (int i = 0; i < size; i++) {
			Preference newPref = new GenericPreference(userID,
					prefs.getItemID(i), prefs.getValue(i));
			if (random.nextDouble() < trainingPercentage) {
				if (oneUserTrainingPrefs == null) {
					oneUserTrainingPrefs = Lists.newArrayListWithCapacity(3);
				}
				oneUserTrainingPrefs.add(newPref);
			} else {
				if (oneUserTestPrefs == null) {
					oneUserTestPrefs = Lists.newArrayListWithCapacity(3);
				}
				oneUserTestPrefs.add(newPref);
			}
		}
		if (oneUserTrainingPrefs != null) {
			trainingPrefs.put(userID, new GenericUserPreferenceArray(
					oneUserTrainingPrefs));
			if (oneUserTestPrefs != null) {
				testPrefs.put(userID, new GenericUserPreferenceArray(
						oneUserTestPrefs));
			}
		}
	 }
	
	 private double getEvaluation(FastByIDMap<PreferenceArray> testPrefs, Recommender recommender)
	    throws TasteException {
	    reset();
	    Collection<Callable<Void>> estimateCallables = Lists.newArrayList();
	    AtomicInteger noEstimateCounter = new AtomicInteger();
	    for (Map.Entry<Long,PreferenceArray> entry : testPrefs.entrySet()) {
	      estimateCallables.add(
	          new PreferenceEstimateCallable(recommender, entry.getKey(), entry.getValue(), noEstimateCounter));
	    }
	    RunningAverageAndStdDev timing = new FullRunningAverageAndStdDev();
	    execute(estimateCallables, noEstimateCounter, timing);
	    return computeFinalEvaluation();
	  }
  
}