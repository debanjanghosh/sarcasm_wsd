package com.wsd.kernel;

import java.util.Collections;
import java.util.HashMap ;
import java.util.Map ;
import java.util.List; 
import java.util.ArrayList ;
import java.util.Set; 

import com.wsd.util.Pair;

//Weiwei code

public class SimWithWE 
{
/*
 * 
 * note: 
 * simliarity3 == just greedy summation and no alignment
 * similarity2 == alignment as in EMNLP paper 
 */
	
	
	public static double th = 0.75;
	
	public static double simliarity3(int[] wi1, double[] val1, int[] wi2, double[] val2) 
	{
	    Map<Double,List<Pair<Integer,Integer>>> scoreMap = new HashMap<Double,List<Pair<Integer,Integer>>>() ;
	    
	    List<Pair<Integer,Integer>> candidates = null ;
	    
	    //no alignment - just add the similarity if the score is really high
	    double totalSim = 0 ;
	    for (int i = 0; i < wi1.length; i++) 
		{
			for (int j = 0; j < wi2.length; j++) 
			{
				Double sim = WordEmbedding.getCosine(wi1[i], wi2[j]);
				if ( sim > 0.85)
				{
					totalSim = totalSim + sim ;
				}
			}
		}
	    return totalSim     ;
	}
	
	public static double simliarity2(int[] wi1, double[] val1, int[] wi2, double[] val2) 
	{
	    Map<Double,List<Pair<Integer,Integer>>> scoreMap = new HashMap<Double,List<Pair<Integer,Integer>>>() ;
	    
	    List<Pair<Integer,Integer>> candidates = null ;
	    
	    for (int i = 0; i < wi1.length; i++) 
		{
			for (int j = 0; j < wi2.length; j++) 
			{
				Double sim = WordEmbedding.getCosine(wi1[i], wi2[j]);
				candidates = scoreMap.get(sim) ;
				if ( null == candidates )
				{
					candidates = new ArrayList<Pair<Integer,Integer>>() ;
				}
				Pair<Integer,Integer> pair = new Pair<Integer, Integer>(i,j);
				candidates.add(pair);
				scoreMap.put(sim, candidates);
			}
		}
		    
	    // choose the alignment greedily
	    //sort the keys in descending order
	    List<Double> keysSorted = new ArrayList<Double>(scoreMap.keySet())  ;
	    java.util.Collections.sort(keysSorted, Collections.reverseOrder());
	    
	    List<Integer>w1chosen = new ArrayList<Integer>();
	    List<Integer>w2chosen = new ArrayList<Integer>();
	    
	    double totalSim = 0 ;
	    double maxSim = -1000 ;
	    
	    for ( Double value : keysSorted)
	    {
	        if ( value >=th)
	        {
	            maxSim = value.doubleValue() ;
	            candidates = scoreMap.get(value) ;
	            
	           //each candidate is above the threshold!
	            for (Pair<Integer,Integer> candidate : candidates )
	            {
	             //  #any element will do, right?
	               int row = candidate.getFirst() ;
	               int column = candidate.getSecond() ;
	           //    #row or column - if anyone is true then we go to the next element
	               if (w1chosen.contains(row)) 
	               {
	                   continue ;
	               }
	               if (w2chosen.contains(column)) 
	               {
	                   continue ;
	               }
	               totalSim = totalSim + maxSim * val1[row] * val2[column] ;
	               
	               w1chosen.add(row) ;
	               w2chosen.add(column) ;
	            }
	        }
	        else
	        {
	        	break ;
	        }
	    }
	    return totalSim     ;
	}
	
	public static double simliarity(int[] wi1, double[] val1, int[] wi2, double[] val2) 
	{
		
		if (wi1.length == 0 || wi2.length == 0)
			return 0;

		if (wi1.length > wi2.length) 
		{ // swap if necessary so that 1st vector is shorter than 2nd vector 
			int[] temp1 = wi1;
			wi1 = wi2;
			wi2 = temp1;
			double[] temp2 = val1;
			val1 = val2;
			val2 = temp2;
		}
		
		// the matrix that contains the pairwise word similarity between two instances
		double[][] S = new double[wi1.length][wi2.length];
		for (int i = 0; i < wi1.length; i++) 
		{
			for (int j = 0; j < wi2.length; j++) 
			{
				S[i][j] = WordEmbedding.getCosine(wi1[i], wi2[j]);
				//System.out.println("i="+i + " j="+j + ":" + S[i][j]);
			}
		}
		
		// choose the alignment greedily
		boolean[] w1chosen = new boolean[wi1.length];
		boolean[] w2chosen = new boolean[wi2.length];
		for (int i = 0; i < wi1.length; i++)
			w1chosen[i] = false;
		for (int j = 0; j < wi2.length; j++)
			w2chosen[j] = false;
		
		double sim = 0;
		double maxSim = 2;
		int index1 = 0, index2 = 0;
		while (true) 
		{
			maxSim = -2;
			for (int i = 0; i < wi1.length; i++) 
			{
				if (w1chosen[i] == true) continue;
				for (int j = 0; j < wi2.length; j++) 
				{
					if (w2chosen[j] == true) continue;
					if (S[i][j] > maxSim) 
					{
						maxSim = S[i][j];
						index1 = i;
						index2 = j;
					}
				}
			}
			if (maxSim > th) 
			{
				w1chosen[index1] = true;
				w2chosen[index2] = true;
				sim += maxSim  * val1[index1] * val2[index2];
				//System.out.println("alignment: i=" + index1 + " j=" + index2);
			}
			else 
			{
				break;
			}
		}
		
		return sim;
	}
}

