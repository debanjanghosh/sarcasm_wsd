package com.wsd.kernel;

import java.io.*;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import libsvm.wsd_node;

//weiwei code

public class WordEmbedding 
{
	public static double[][] we;
	
	public static HashMap<String,Integer> vocab;
	
	public  void readGloveWE(String filename) 
	{
		vocab = new HashMap<String,Integer>();
		try
		{
			BufferedReader br = new BufferedReader(new FileReader(filename));
			// find the number of dimensions and words
			String line = br.readLine();
			line = line.trim();
			int K = line.split("\\s+").length-1;
			int numWords = 1;
			while ((line = br.readLine()) != null) 
			{
				numWords++;
			}
			
			br.close();
			System.out.println("number of dimensions: " + K);
			System.out.println("number of words: " + numWords);

			br = new BufferedReader(new FileReader(filename));
			we = new double[numWords][K];
			int i = 0;
			while ((line = br.readLine()) != null) {
				line = line.trim();
				String[] info = line.split("\\s+");
				vocab.put(info[0], i);
				for (int j = 0; j < K; j++)
					we[i][j] = Double.parseDouble(info[j+1]);
				i++;
			}
			//System.out.println("index of 'of' = " + vocab.get("of"));
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public  void readGensimWE(String filename) 
	{
		vocab = new HashMap<String,Integer>();
		try
		{
			BufferedReader br = new BufferedReader(new FileReader(filename));
			// find the number of dimensions and words
			String line = br.readLine();
			line = line.trim();
			int K = line.split("\\s+").length-1;
			int numWords = 1;
			while ((line = br.readLine()) != null) 
			{
				numWords++;
			}
			
			br.close();
			System.out.println("number of dimensions: " + K);
			System.out.println("number of words: " + numWords);

			br = new BufferedReader(new FileReader(filename));
			we = new double[numWords+1][K]; //+1 because the first, that is zeroth is the unknown vector ---
			java.util.Random rng = new java.util.Random();
			for(int i = 0;i<K;i++) 
			{
			      we[0][i] = rng.nextGaussian();
			}
			
			int i = 1;
			while ((line = br.readLine()) != null) 
			{
				line = line.trim();
				String[] info = line.split("\\s+");
				vocab.put(info[0], i);
				for (int j = 0; j < K; j++)
					we[i][j] = Double.parseDouble(info[j+1]);
				i++;
			}
			//System.out.println("index of 'of' = " + vocab.get("of"));
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public  void readWord2VecWE(String filename) 
	{
		vocab = new HashMap<String,Integer>();
		try
		{
			BufferedReader br = new BufferedReader(new FileReader(filename));
			// find the number of dimensions and words
			String line = br.readLine();
			line = line.trim();
			int K = Integer.valueOf(line.split("\\s+")[1]) ;
			int numWords = Integer.valueOf(line.split("\\s+")[0]) ;
			System.out.println("number of dimensions: " + K);
			System.out.println("number of words: " + numWords);

			br = new BufferedReader(new FileReader(filename));
			we = new double[numWords+1][K]; //+1 because the first, that is zeroth is the unknown vector ---
			java.util.Random rng = new java.util.Random();
			for(int i = 0;i<K;i++) 
			{
			      we[0][i] = rng.nextGaussian();
			}
			
			int i = 1;
			String header = br.readLine() ;
			while ((line = br.readLine()) != null) 
			{
				line = line.trim();
				String[] info = line.split("\\s+");
				vocab.put(info[0], i);
				for (int j = 0; j < K; j++)
					we[i][j] = Double.parseDouble(info[j+1]);
				i++;
			}
			//System.out.println("index of 'of' = " + vocab.get("of"));
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		System.out.println("finished loading the embeddings") ;
	}

	
	public  void readGensimWENoUnknow(String filename) 
	{
		vocab = new HashMap<String,Integer>();
		try
		{
			BufferedReader br = new BufferedReader(new FileReader(filename));
			// find the number of dimensions and words
			String line = br.readLine();
			line = line.trim();
			int K = line.split("\\s+").length-1;
			int numWords = 1;
			while ((line = br.readLine()) != null) 
			{
				numWords++;
			}
			
			br.close();
			System.out.println("number of dimensions: " + K);
			System.out.println("number of words: " + numWords);

			br = new BufferedReader(new FileReader(filename));
			we = new double[numWords][K];
			int i = 0;
			while ((line = br.readLine()) != null) 
			{
				line = line.trim();
				String[] info = line.split("\\s+");
				vocab.put(info[0], i);
				for (int j = 0; j < K; j++)
					we[i][j] = Double.parseDouble(info[j+1]);
				i++;
			}
			//System.out.println("index of 'of' = " + vocab.get("of"));
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	
	
	/*
	 * Returns a similarity score 
	 */
	public static double getCosine(int w1, int w2) 
	{
		if (w1 == w2) 
		{
			return 1; // if the index are the same, return maximum sim score
		}
		//old function
	/*	
		if (w1 >= we.length || w2 >= we.length) // if one word does not have an embedding
		{	
			return 0;
		}
	*/	
		//new function
		if (w1 >= we.length)
		{
			w1 = 0 ;
		}
		if (w2 >= we.length)
		{
			w2 = 0 ;
		}
		
		
		double sim = 0;
		double len1 = 0, len2 = 0;
		for (int i = 0; i < we[w1].length; i++) {
			sim += we[w1][i] * we[w2][i];
			len1 += we[w1][i] * we[w1][i];
			len2 += we[w2][i] * we[w2][i];
		}
		len1 = Math.sqrt(len1);
		len2 = Math.sqrt(len2);
		if (len1 == 0 || len2 == 0)
			return 0;
		sim /= len1*len2;
		return sim;
	}
	
	//interface for libSVM modified code 
	public double wsdKernelInterface(wsd_node[] one, wsd_node[] two)
	{
		
		Object[] array = WordEmbedding.indexSentence(one);
		int[] wi1 = (int[]) array[0];
		double[] val1 = (double[]) array[1];

		array = WordEmbedding.indexSentence(two);
		int[] wi2 = (int[]) array[0];
		double[] val2 = (double[]) array[1];
	
		//similarity2 = greedy alignment, similarity3 = just add the total similarity
		return SimWithWE.simliarity2(wi1, val1, wi2, val2);
	//	return SimWithWE.simliarity3(wi1, val1, wi2, val2);
		
	
	}
	
	/*
	 * Index a sentence, and return two arrays: one stores word index, and the other stores values
	 * also remove the target
	 */
	public static Object[] indexSentence(wsd_node[] words) 
	{
		HashMap<Integer,Double> sent = new HashMap<Integer,Double>();
		
		for (int j = 0; j < words.length; j++) 
		{
			//System.out.print("word=" + words[i]);
			// if the word does not have an embedding, create an index
			//we filter the target work - we do not use it in calculation
			wsd_node node = words[j] ;
			if ( null == node )
			{
				continue ;
			}
			if ( null == words[j].token )
			{
				continue ;
			}
			String word =  words[j].token  ;
			
			if (!vocab.containsKey(word))
				vocab.put(word, vocab.size());
			int w = vocab.get(word);
			double v = 1;
			if (sent.containsKey(w))
				v += sent.get(w);
			sent.put(w,v);
			//System.out.print(" wi=" + w + " val=" + v);
			//System.out.println();
		}
		int[] wi = new int[sent.size()];
		double[] val = new double[sent.size()];
		int i = 0;
		for (Map.Entry<Integer,Double> me : sent.entrySet()) {
			wi[i] = me.getKey();
			val[i] = me.getValue();
			//System.out.print(wi[i] + ":" +val[i] + " ");
			i++;
		}
		//System.out.println();
		return new Object[]{wi, val};
	}
	
	
	/*
	 * Index a sentence, and return two arrays: one stores word index, and the other stores values
	 * also remove the target
	 */
	public static Object[] indexSentence(String str, String target) 
	{
		String[] words = str.trim().split("\\s+");
		HashMap<Integer,Double> sent = new HashMap<Integer,Double>();
		
		for (int i = 0; i < words.length; i++) 
		{
			//System.out.print("word=" + words[i]);
			// if the word does not have an embedding, create an index
			//we filter the target work - we do not use it in calculation
			if (words[i].equalsIgnoreCase(target) || words[i].equalsIgnoreCase("#"+target))
			{
				continue ;
			}
			
			
			if (!vocab.containsKey(words[i]))
				vocab.put(words[i], vocab.size());
			int w = vocab.get(words[i]);
			double v = 1;
			if (sent.containsKey(w))
				v += sent.get(w);
			sent.put(w,v);
			//System.out.print(" wi=" + w + " val=" + v);
			//System.out.println();
		}
		int[] wi = new int[sent.size()];
		double[] val = new double[sent.size()];
		int i = 0;
		for (Map.Entry<Integer,Double> me : sent.entrySet()) {
			wi[i] = me.getKey();
			val[i] = me.getValue();
			//System.out.print(wi[i] + ":" +val[i] + " ");
			i++;
		}
		//System.out.println();
		return new Object[]{wi, val};
	}
	
	/*
	 * Index a sentence, and return two arrays: one stores word index, and the other stores values
	 * also remove the target
	 */
	public static Object[] indexList(List<String> words, String target) 
	{
	//	String[] words = str.trim().split("\\s+");
		HashMap<Integer,Double> sent = new HashMap<Integer,Double>();
		
		for (int i = 0; i < words.size(); i++) 
		{
			//System.out.print("word=" + words[i]);
			// if the word does not have an embedding, create an index
			//we filter the target work - we do not use it in calculation
			if (words.get(i).equalsIgnoreCase(target) || words.get(i).equalsIgnoreCase("#"+target))
			{
				continue ;
			}
			
			
			if (!vocab.containsKey(words.get(i)))
				vocab.put(words.get(i), vocab.size());
			int w = vocab.get(words.get(i));
			double v = 1;
			if (sent.containsKey(w))
				v += sent.get(w);
			sent.put(w,v);
			//System.out.print(" wi=" + w + " val=" + v);
			//System.out.println();
		}
		int[] wi = new int[sent.size()];
		double[] val = new double[sent.size()];
		int i = 0;
		for (Map.Entry<Integer,Double> me : sent.entrySet()) 
		{
			wi[i] = me.getKey();
			val[i] = me.getValue();
			//System.out.print(wi[i] + ":" +val[i] + " ");
			i++;
		}
		//System.out.println();
		return new Object[]{wi, val};
	}
	
	
	/*
	 * Index a sentence, and return two arrays: one stores word index, and the other stores values
	 */
	public synchronized static Object[] indexSentence(String str) 
	{
		String[] words = str.trim().split("\\s+");
		HashMap<Integer,Double> sent = new HashMap<Integer,Double>();
		
		for (int i = 0; i < words.length; i++) 
		{
			//System.out.print("word=" + words[i]);
			// if the word does not have an embedding, create an index
			if (!vocab.containsKey(words[i]))
				vocab.put(words[i], vocab.size());
			int w = vocab.get(words[i]);
			double v = 1;
			if (sent.containsKey(w))
				v += sent.get(w);
			sent.put(w,v);
			//System.out.print(" wi=" + w + " val=" + v);
			//System.out.println();
		}
		int[] wi = new int[sent.size()];
		double[] val = new double[sent.size()];
		int i = 0;
		for (Map.Entry<Integer,Double> me : sent.entrySet()) {
			wi[i] = me.getKey();
			val[i] = me.getValue();
			//System.out.print(wi[i] + ":" +val[i] + " ");
			i++;
		}
		//System.out.println();
		return new Object[]{wi, val};
	}
	
	
	
	public static void main(String[] args) 
	{
	//	String filename = "./data/glove/input/glove.twitter.27B.25d.txt";
		String filename = "./data/config/tweet.all.03222015.cbow.model.bin.txt";
		
		WordEmbedding obj = new WordEmbedding() ;
	//	obj.readGloveWE(filename);
		obj.readGensimWE(filename);


		String str = "the . , . gww";
		str = "the world is mysterious and a crazy place no where to go:)";
		Object[] array = WordEmbedding.indexSentence(str);
		int[] wi1 = (int[]) array[0];
		double[] val1 = (double[]) array[1];

		String str2 = "i think the , gww";
		str2 = "strange world it is and a mad mad mad world where to hide ;)" ;
		array = WordEmbedding.indexSentence(str2);
		int[] wi2 = (int[]) array[0];
		double[] val2 = (double[]) array[1];
		
		double s = SimWithWE.simliarity(wi1, val1, wi2, val2);
		double s11 = SimWithWE.simliarity(wi1, val1, wi1, val1);
		double s22 = SimWithWE.simliarity(wi2, val2, wi2, val2);
		System.out.println("sim=" + s);
		s = s/Math.sqrt(s11*s11 + s22 * s22 );
		System.out.println("norm sim=" + s);
		
		s = SimWithWE.simliarity2(wi1, val1, wi2, val2);
		s11 = SimWithWE.simliarity2(wi1, val1, wi1, val1);
		s22 = SimWithWE.simliarity2(wi2, val2, wi2, val2);
		System.out.println("sim=" + s);
		s = s/Math.sqrt(s11*s11 + s22 * s22 );
		System.out.println("norm sim=" + s);

	}
}
