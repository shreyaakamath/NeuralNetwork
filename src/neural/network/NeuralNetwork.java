package neural.network;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

class DataInstance{
	int id;
	List<Double> attValues;
	String givenClass;
}

class SortedDi{
	int id;
	String output;
	
	SortedDi(int id,String output){
		this.id=id;
		this.output=output;
	}
}

class SortedDiComparator implements Comparator<SortedDi>{
	public int compare(SortedDi d1,SortedDi d2){
		if(d1.id<d2.id) return -1;
		else if (d1.id>d2.id) return 1;
		else return 0;
	}
}
public class NeuralNetwork {
	int noAttr;
	String[] classValues;
	int folds;
	int epoch;
	double learningRate;
	List<DataInstance> allData;
	List<DataInstance> posData;
	List<DataInstance> negData;
	List<List<DataInstance>> cvData;
	List<Double> NnWeight;
	
	static double threshold=0.5;
	static double initialWeight=0.1;
	static int[] classOpValues = {0,1};
	
	NeuralNetwork(String fileName,double learningRate,int folds,int epoch){
		this.learningRate=learningRate;
		this.folds=folds;
		this.epoch=epoch;
		allData = new ArrayList<DataInstance>();
		posData = new ArrayList<DataInstance>();
		negData = new ArrayList<DataInstance>();
		cvData= new ArrayList<List<DataInstance>>();
		NnWeight=new ArrayList<Double>();
		BufferedReader reader=null; 
		Instances data=null;
		
		try {
			reader = new BufferedReader(new FileReader(fileName));
			data = new Instances(reader);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} 
		//get the number of attributes and the class attribute values
		noAttr=data.numAttributes();
		Attribute attr=data.attribute(noAttr-1);
		if(attr.name().equalsIgnoreCase("class")){
			String attrRegex="\\{(.*?)\\}";
			Pattern p2 = Pattern.compile(attrRegex);
			Matcher m2 = p2.matcher(attr.toString());
		    m2.find();
		    String att=m2.group(1);
		    classValues = att.split(",");
		}
		
		//populate the initial weight in the NN
		for(int i=0;i<noAttr;i++) NnWeight.add(initialWeight);
		
		//read in the training data instances. create allData,posData,negData.
		for (int i = 0;i < data.numInstances();i++){
			DataInstance di=new DataInstance();
			Instance inst = data.instance(i);
			String[] items=inst.toString().split(",");
			di.attValues= new ArrayList<Double>();
			for(int j=0;j<noAttr-1;j++){
				di.attValues.add(new Double(items[j]));
			}
			di.givenClass=items[noAttr-1];
			di.id=i;
			allData.add(di);
			if(di.givenClass.equalsIgnoreCase(classValues[0])) posData.add(di);
			else negData.add(di);
		}
		
		
		//Distribute the positive and negative samples proportionally to create stratified cv data
		for(int i=0;i<folds;i++){
			cvData.add(new ArrayList<DataInstance>());
		}
		
		int index=0;
		for(DataInstance di: posData){
			cvData.get(index).add(di);
			index=(index+1)%(folds-1);
		}
		for(DataInstance di: negData){
			cvData.get(index).add(di);
			index=(index+1)%(folds-1);
		}
	}

	/**
	 * For one DataInstance , train the NN epoch number of times
	 */
	void NNTrain(DataInstance di){
		for(int j=0;j<epoch;j++){
			//calculate wi*xi for all attributes
			double sum=0;
			for(int i=0;i<noAttr-1;i++){
				sum+=di.attValues.get(i)*NnWeight.get(i);
			}
			
			//calculate 1/(1+e^(-sum)
			double denominator=(double)1+Math.exp(-1*sum);
			double oj=(double)1/denominator;
			
			int yj=di.givenClass.equalsIgnoreCase(classValues[0])?classOpValues[0]:classOpValues[1];
			
			//calculate delta oj*(1-oj)*(yj-oj)
			double delta=oj*(1-oj)*(yj-oj);
			
			//calculate delta weight and update all weights
			for(int i=0;i<noAttr-1;i++){
				double deltaWeight=delta*learningRate*di.attValues.get(i);
				NnWeight.set(i,deltaWeight+NnWeight.get(i));
			}
		}
	}
	
	/**
	 * For one DataInstance , predict the class based on trained NN.
	 */
	void NNTest(int fold,DataInstance di,List<SortedDi> output){
		//calculate wi*xi for all attributes
		double sum=0;
		for(int i=0;i<noAttr-1;i++){
			sum+=di.attValues.get(i)*NnWeight.get(i);
		}
		
		//calculate 1/(1+e^(-sum)
		double denominator=(double)1+Math.exp(-1*sum);
		double sigmoid=(double)1/denominator;
		
		String predictedClass;
		String actualClass=di.givenClass;
		NumberFormat formatter = new DecimalFormat("#0.############");
		
		if(sigmoid>0.5) predictedClass=classValues[0];
		else predictedClass=classValues[1];
		
		int index=di.id;
		String op=fold+" "+predictedClass+" "+actualClass+" "+formatter.format(sigmoid);
		output.add(new SortedDi(index, op));
	}
	
	/**
	 * Use the generated Cv data to perform CV and train/test the NN
	 */
	void stratifiedCrossValidation(){
		List<DataInstance> train= new ArrayList<DataInstance>();
		List<DataInstance> test= new ArrayList<DataInstance>();
		List<SortedDi> output= new ArrayList<SortedDi>();
		for(int i=0;i<folds;i++){
			for(int j=0;j<folds;j++){
				if(j==i){
					//this is test data
					test.addAll(cvData.get(j));
				}else{
					train.addAll(cvData.get(i));
				}
			}
			long seed=System.nanoTime();
			Collections.shuffle(train,new Random(seed));
			Collections.shuffle(test,new Random(seed));
			for(DataInstance di:train) NNTrain(di);
			for(DataInstance di:test) NNTest(i+1, di,output);
		}
		Collections.sort(output,new SortedDiComparator());
		for(SortedDi di:output) System.out.println(di.output);
	}
	void validate(){
		System.out.println("====pos=====");	
		for(DataInstance i:posData) System.out.println(i.givenClass);
		System.out.println("====neg=====");
		for(DataInstance i:negData) System.out.println(i.givenClass);
	}
	
	public static void main(String[] args) {
		NeuralNetwork nn=new NeuralNetwork("sonar.arff", 10, 10, 1);
//		nn.NNTrain(nn.allData.get(0));
//		nn.NNTest(1,nn.allData.get(0));
		nn.stratifiedCrossValidation();
//		nn.validate();
	}

}
