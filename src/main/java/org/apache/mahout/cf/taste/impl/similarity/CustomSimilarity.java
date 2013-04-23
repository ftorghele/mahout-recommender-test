package org.apache.mahout.cf.taste.impl.similarity;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.HashMap;
import java.util.Scanner;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.common.Weighting;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.impl.similarity.AbstractSimilarity;

import com.google.common.base.Preconditions;

public final class CustomSimilarity extends AbstractSimilarity {

  private HashMap<Long, Integer> userAgeMap = new HashMap<Long, Integer>();
  private HashMap<Long, Boolean> userMaleMap = new HashMap<Long, Boolean>();

  public CustomSimilarity(DataModel dataModel) throws TasteException {
    this(dataModel, Weighting.UNWEIGHTED);
  }

  public CustomSimilarity(DataModel dataModel, Weighting weighting) throws TasteException {	
    super(dataModel, weighting, true);
    Preconditions.checkArgument(dataModel.hasPreferenceValues(), "DataModel doesn't have preference values");
  
    File userMapFile = new File("datasets/movielens/ml-100k/users.csv");
	Scanner scan = null;
	try {
		scan = new Scanner(userMapFile);
	} catch (FileNotFoundException e) {
		e.printStackTrace();
	}
	while(scan.hasNextLine()){
		String[] line = scan.nextLine().split(",");
		userAgeMap.put(Long.parseLong(line[0]), Integer.parseInt(line[1]));
		if(line[2].equals("M")) {
			userMaleMap.put(Long.parseLong(line[0]), true);
		} else {
			userMaleMap.put(Long.parseLong(line[0]), false);
		}
	}
	scan.close();
  }
  
  @Override
  public double userSimilarity(long userID1, long userID2) throws TasteException {
	  double result = super.userSimilarity(userID1, userID2);
	  
	  if(userMaleMap.get(userID1) == userMaleMap.get(userID2)) {
		  result += 0.1;
	  } else {
		  result -= 0.1;
	  }
	  
	  if(userAgeMap.get(userID1) == userAgeMap.get(userID2)) {
		  result += 0.1;
	  } else {
		  result -= 0.1;
	  }
	  
	  result = (result > 1)? 1 : result;
	  result = (result < -1)? -1 : result;
	    
	  return result;
  }
  
  @Override
  double computeResult(int n, double sumXY, double sumX2, double sumY2, double sumXYdiff2) {
    if (n == 0) {
      return Double.NaN;
    }
    // Note that sum of X and sum of Y don't appear here since they are assumed to be 0;
    // the data is assumed to be centered.
    double denominator = Math.sqrt(sumX2) * Math.sqrt(sumY2);
    if (denominator == 0.0) {
      // One or both parties has -all- the same ratings;
      // can't really say much similarity under this measure
      return Double.NaN;
    }
    return sumXY / denominator;
  }
  
}
