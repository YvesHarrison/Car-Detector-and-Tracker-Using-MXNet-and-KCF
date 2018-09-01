import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.BufferedReader;
public class invoke{
    public static void main(String args[])throws IOException,InterruptedException{
        
        try{
            Process proc= Runtime.getRuntime().exec("python mul_q.py ./test_clip.avi");
            proc.waitFor();  
        }catch(IOException e){
            e.printStackTrace();
        }
        
        /*Process proc = null;
        try {
            proc = Runtime.getRuntime().exec("python 2.py");
            proc.waitFor();
            InputStreamReader isr = new InputStreamReader(proc.getInputStream());
        char[] temp = new char[20];
        isr.read(temp);
            System.out.println(temp);

        }catch(IOException e){
            e.printStackTrace();
        }*/
    }
}