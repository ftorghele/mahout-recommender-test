package org.apache.mahout.cf.taste.impl.eval;

import java.util.Collection;
import java.util.List;
import java.util.Map;
//import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;

import com.google.common.collect.Lists;
import org.apache.mahout.cf.taste.common.NoSuchItemException;
import org.apache.mahout.cf.taste.common.NoSuchUserException;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.DataModelBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverageAndStdDev;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.impl.common.RunningAverageAndStdDev;
import org.apache.mahout.cf.taste.impl.model.GenericPreference;
import org.apache.mahout.cf.taste.impl.model.GenericUserPreferenceArray;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.Recommender;
//import org.apache.mahout.common.RandomUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CustomEvaluator {

	private static final Logger log = LoggerFactory
			.getLogger(CustomEvaluator.class);

	private float maxPreference;
	private float minPreference;
//	private final Random random;

	private RunningAverage average;

	public CustomEvaluator() {
//		random = RandomUtils.getRandom();
		maxPreference = Float.NaN;
		minPreference = Float.NaN;
	}

	public final float getMaxPreference() {
		return maxPreference;
	}

	public final void setMaxPreference(float maxPreference) {
		this.maxPreference = maxPreference;
	}

	public final float getMinPreference() {
		return minPreference;
	}

	public final void setMinPreference(float minPreference) {
		this.minPreference = minPreference;
	}

	public double evaluate(RecommenderBuilder recommenderBuilder,
			DataModelBuilder dataModelBuilder, DataModel trainingModel,
			DataModel testModel) throws TasteException {

		FastByIDMap<PreferenceArray> testPrefs = new FastByIDMap<PreferenceArray>(
				1 + (int) (testModel.getNumUsers()));


		LongPrimitiveIterator it = trainingModel.getUserIDs();
		while (it.hasNext()) {
			long userID = it.nextLong();

			List<Preference> userTestPrefsList = null;
			PreferenceArray userTestPrefs = testModel
					.getPreferencesFromUser(userID);

			for (int i = 0; i < userTestPrefs.length(); i++) {
				Preference newPref = new GenericPreference(userID,
						userTestPrefs.getItemID(i), 
						userTestPrefs.getValue(i));
				if (userTestPrefsList == null) {
					userTestPrefsList = Lists.newArrayListWithCapacity(3);
				}
				userTestPrefsList.add(newPref);
			}

			if (trainingModel.getPreferencesFromUser(userID).length() > 0) {

				if (userTestPrefsList != null) {
					testPrefs.put(userID, new GenericUserPreferenceArray(
							userTestPrefsList));
				}
			}
		}

		Recommender recommender = recommenderBuilder
				.buildRecommender(trainingModel);

		double result = getEvaluation(testPrefs, recommender);
		log.info("Evaluation result: {}", result);
		return result;
	}

	private float capEstimatedPreference(float estimate) {
		if (estimate > maxPreference) {
			return maxPreference;
		}
		if (estimate < minPreference) {
			return minPreference;
		}
		return estimate;
	}

	private double getEvaluation(FastByIDMap<PreferenceArray> testPrefs,
			Recommender recommender) throws TasteException {
		reset();
		Collection<Callable<Void>> estimateCallables = Lists.newArrayList();
		AtomicInteger noEstimateCounter = new AtomicInteger();
		for (Map.Entry<Long, PreferenceArray> entry : testPrefs.entrySet()) {
			estimateCallables.add(new PreferenceEstimateCallable(recommender,
					entry.getKey(), entry.getValue(), noEstimateCounter));
		}
		log.info("Beginning evaluation of {} users", estimateCallables.size());
		RunningAverageAndStdDev timing = new FullRunningAverageAndStdDev();
		execute(estimateCallables, noEstimateCounter, timing);
		return computeFinalEvaluation();
	}

	protected static void execute(Collection<Callable<Void>> callables,
			AtomicInteger noEstimateCounter, RunningAverageAndStdDev timing)
			throws TasteException {

		callables = wrapWithStatsCallables(callables, noEstimateCounter, timing);
		int numProcessors = Runtime.getRuntime().availableProcessors();
		ExecutorService executor = Executors.newFixedThreadPool(numProcessors);
		log.info("Starting timing of {} tasks in {} threads", callables.size(),
				numProcessors);
		try {
			List<Future<Void>> futures = executor.invokeAll(callables);
			// Go look for exceptions here, really
			for (Future<Void> future : futures) {
				future.get();
			}
		} catch (InterruptedException ie) {
			throw new TasteException(ie);
		} catch (ExecutionException ee) {
			throw new TasteException(ee.getCause());
		}
		executor.shutdown();
	}

	private static Collection<Callable<Void>> wrapWithStatsCallables(
			Iterable<Callable<Void>> callables,
			AtomicInteger noEstimateCounter, RunningAverageAndStdDev timing) {
		Collection<Callable<Void>> wrapped = Lists.newArrayList();
		int count = 0;
		for (Callable<Void> callable : callables) {
			boolean logStats = count++ % 1000 == 0; // log every 1000 or so
													// iterations
			wrapped.add(new StatsCallable(callable, logStats, timing,
					noEstimateCounter));
		}
		return wrapped;
	}

	protected void reset() {
		average = new FullRunningAverage();
	}

	protected void processOneEstimate(float estimatedPreference,
			Preference realPref) {
		double diff = realPref.getValue() - estimatedPreference;
		average.addDatum(diff * diff);
	}

	protected double computeFinalEvaluation() {
		return Math.sqrt(average.getAverage());
	}

	public String toString() {
		return "CustomEvaluator";
	}

	public final class PreferenceEstimateCallable implements Callable<Void> {

		private final Recommender recommender;
		private final long testUserID;
		private final PreferenceArray prefs;
		private final AtomicInteger noEstimateCounter;

		public PreferenceEstimateCallable(Recommender recommender,
				long testUserID, PreferenceArray prefs,
				AtomicInteger noEstimateCounter) {
			this.recommender = recommender;
			this.testUserID = testUserID;
			this.prefs = prefs;
			this.noEstimateCounter = noEstimateCounter;
		}

		@Override
		public Void call() throws TasteException {
			for (Preference realPref : prefs) {
				float estimatedPreference = Float.NaN;
				try {
					estimatedPreference = recommender.estimatePreference(
							testUserID, realPref.getItemID());
				} catch (NoSuchUserException nsue) {
					// It's possible that an item exists in the test data but
					// not training data in which case
					// NSEE will be thrown. Just ignore it and move on.
					log.info(
							"User exists in test data but not training data: {}",
							testUserID);
				} catch (NoSuchItemException nsie) {
					log.info(
							"Item exists in test data but not training data: {}",
							realPref.getItemID());
				}
				if (Float.isNaN(estimatedPreference)) {
					noEstimateCounter.incrementAndGet();
				} else {
					estimatedPreference = capEstimatedPreference(estimatedPreference);
					processOneEstimate(estimatedPreference, realPref);
				}
			}
			return null;
		}

	}
}