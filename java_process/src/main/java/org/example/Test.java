package org.example;

import com.github.gumtreediff.actions.ChawatheScriptGenerator;
import com.github.gumtreediff.actions.EditScript;
import com.github.gumtreediff.actions.EditScriptGenerator;
import com.github.gumtreediff.actions.SimplifiedChawatheScriptGenerator;
import com.github.gumtreediff.actions.model.Action;
import com.github.gumtreediff.client.*;
import com.github.gumtreediff.gen.TreeGenerators;
import com.github.gumtreediff.matchers.MappingStore;
import com.github.gumtreediff.matchers.Matcher;
import com.github.gumtreediff.matchers.Matchers;
import com.github.gumtreediff.tree.Tree;

import org.json.JSONArray;
import org.json.JSONObject;
import java.io.FileWriter;
import java.io.IOException;
import java.io.File;
import java.util.Objects;

public class Test {
    public static void main(String[] args) throws IOException {
        Run.initGenerators(); // registers the available parsers
        String src_root_path = "/home/shweng/code_data_clean/code-refinement/data/small-addMain/buggy-test/";
        String dst_root_path = "/home/shweng/code_data_clean/code-refinement/data/small-addMain/fixed-test/";
        String output_path = "/home/shweng/code_data_clean/code-refinement/data/small-addMain/test-action.json";
        File src_directory = new File(src_root_path);
        File dst_directory = new File(dst_root_path);
        File[] src_files = src_directory.listFiles();
        File[] dst_files = dst_directory.listFiles();
        int length = src_files.length;
        assert length == dst_files.length;
        System.out.println("length: " + length);
        JSONArray jsonActions = new JSONArray();
        for (int i = 0; i < length; i++) {
            // progress bar

            String srcFile = src_files[i].getAbsolutePath();
            String dstFile = dst_files[i].getAbsolutePath();
            System.out.println("Processing file " + i + " of " + length+ ":" + srcFile);
            Tree src = TreeGenerators.getInstance().getTree(srcFile).getRoot(); // retrieves and applies the default parser for the file
            Tree dst = TreeGenerators.getInstance().getTree(dstFile).getRoot(); // retrieves and applies the default parser for the file

            Matcher defaultMatcher = Matchers.getInstance().getMatcher(); // retrieves the default matcher
            MappingStore mappings = defaultMatcher.match(src, dst); // computes the mappings between the trees
            EditScriptGenerator editScriptGenerator = new ChawatheScriptGenerator(); // instantiates the simplified Chawathe script generator
            EditScript actions = editScriptGenerator.computeActions(mappings); // computes the edit script

            // rev
            Tree src2 = TreeGenerators.getInstance().getTree(srcFile).getRoot(); // retrieves and applies the default parser for the file
            Tree dst2 = TreeGenerators.getInstance().getTree(dstFile).getRoot();
            Matcher defaultMatcher2 = Matchers.getInstance().getMatcher(); // retrieves the default matcher
            MappingStore mappings2 = defaultMatcher2.match(dst2, src2); // computes the mappings between the trees
            EditScriptGenerator editScriptGenerator2 = new ChawatheScriptGenerator(); // instantiates the simplified Chawathe script generator
            EditScript actions_rev = editScriptGenerator2.computeActions(mappings2); // computes the edit script

            JSONObject actionJson = new JSONObject();
            // array of actions
            JSONArray opts = new JSONArray();
            JSONArray fulls = new JSONArray();
            for (Action action : actions) {
                opts.put(action.getName());
                fulls.put(action.toString());
            }
            // array of actions rev
            JSONArray opts_rev = new JSONArray();
            JSONArray fulls_rev = new JSONArray();
            for (Action action : actions_rev) {
                opts_rev.put(action.getName());
                fulls_rev.put(action.toString());
            }

            // idx is the java file name without the extension
            int idx = Integer.parseInt(src_files[i].getName().split("\\.")[0]);
            actionJson.put("idx", String.valueOf(idx));
            actionJson.put("opts", opts);
            actionJson.put("fulls", fulls);
            actionJson.put("opts_rev", opts_rev);
            actionJson.put("fulls_rev", fulls_rev);
            jsonActions.put(actionJson);
        }
        try (FileWriter file = new FileWriter(output_path)) {
            file.write(jsonActions.toString(4)); // pretty print with an indent of 4
        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}