/**
 * 
 * 
 *
 * @author ibrahim AKIN
 *		   
 */

class DynamicArray{
	
	private int n, capacity; 
	private int A[];
					
	
	private int e = 0, s = 0, eb = 0, sb = 0, t = 0;   
	
	public DynamicArray() {
		this.n = 0;
		this.capacity = 1;           
		this.A = new int[capacity];  
	}
	
	
	
	
	
	
	
	
	
	
	
	public void Append(int a) {
		e++;
		if(n >= capacity) {        
			resize(2 * capacity);  
			
			eb++;
			System.out.println(" Dizi büyütüldü.\n" + getInfo(n+"+1") + "\n");
		}
		A[n] = a;
		n++;
	}
	
	public void Remove() {
		if(n > 0) {
			A[n-1] = 0;
			n--;
			
			s++;
			if(capacity > 1 && n < capacity / 2 ) {  
				resize(capacity / 2);                
				
				sb++;
				System.out.println(" Dizi küçültüldü.\n" + getInfo(n+"") + "\n");
			}
		}
	}
	
	private void resize(int c) {
		t = 0;
		int B[] = new int[c];			
		for (int i = 0; i < n; i++) {	
			B[i] = A[i];				
										
			t++;
		}
		A = B;
		capacity = c;
	}
	
	@Override
	public String toString() {
		String temp = "";
		for (int i = 0; i < n; i++) {
			temp += " " + A[i];
		}
		return temp;
	}
	
	public String getInfo() {
		return ("Eleman Sayýsý = " + n + ", Kapasite = " + capacity + 
				",\nBoyut Deðiþikliði Ekle + Sil = " + eb + " + " + sb + " = " + (eb+sb) +
				",\n           Toplam Ekle + Sil = " + e + " + " + s + " = " + (e+s));
	}
	
	private String getInfo(String n) {
		return ("Eleman Sayýsý = " + n + ", Kapasite = " + capacity + ",\nTaþýma Sayýsý = " + t );
	}
}


public class AA_odev02_ibrahimAKIN {
	
	static void sonaEkle(DynamicArray dynamicArray, int a, int n) {
		
		for (int i = 0; i < n; i++) {
			dynamicArray.Append(a);
		}
		
	}
	
	static void sondanSil(DynamicArray dynamicArray, int n) {
		
		for (int i = 0; i < n; i++) {
			dynamicArray.Remove();
		}
		
	}
	
	public static void main(String[] args) {
		
		
		
		
		
		
		DynamicArray dynamicArray = new DynamicArray();
		System.out.println(" Dizinin ilk durumu:\n" + dynamicArray.getInfo() + "\n");
		
		
		
		sonaEkle(dynamicArray, 5, 10);     
		
		System.out.println(dynamicArray.getInfo() + "\n\n" + dynamicArray.toString() + "\n");
		
		
		
		sondanSil(dynamicArray, 5);        
		
		System.out.println(dynamicArray.getInfo() + "\n\n" + dynamicArray.toString() + "\n");
		
		
		
		sondanSil(dynamicArray, 7);//(5) 
		
		System.out.println(" Dizinin son durumu:\n" + dynamicArray.getInfo() + "\n\n\n*\n");
		
		
		
		sonaEkle(dynamicArray, 5, 1);
		sonaEkle(dynamicArray, -100, 1);
		System.out.println("\n Dizinin son durumu:\n" + dynamicArray.getInfo() + "\n\n");
		
		sondanSil(dynamicArray, 1);
		sondanSil(dynamicArray, 1);
		sondanSil(dynamicArray, 1);
		System.out.println("\n Dizinin son durumu:\n" + dynamicArray.getInfo() + "\n\n");
		
		sonaEkle(dynamicArray, 10, 2);
		sonaEkle(dynamicArray, -4, 4);
		System.out.println("\n Dizinin son durumu:\n" + dynamicArray.getInfo() + "\n\n");
	}
}