import java.io.IOException;
import java.util.*;
import org.python.core.Py;
import org.python.core.PyException;
import org.python.core.PyFunction;
import org.python.core.PyObject;
import org.python.util.PythonInterpreter;
import org.python.core.PyString;
import org.python.core.PyList;
import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;
public class pij{
    public static void main(String args[]){
        PythonInterpreter interpreter=new PythonInterpreter();
        PyList pylist = new PyList();
        for (int i=0;i<args.length;++i){
            PyObject ob=new PyString(args[i].toString());
            pylist.append(ob);
        }
        interpreter.execfile("./mul_f.py");
        PyFunction func=(PyFunction) interpreter.get("video_trackinga",PyFunction.class);
        PyList obj=func._call_(pylist);//report error that pyfunction does not have symbol _call_
        /*interpreter.exec("day=(1,2,3,4,5,6,7);");
        interpreter.exec("print day[1];");*/
        
    }
}