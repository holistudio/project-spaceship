using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using System.Linq;

using System.IO;
using System.Text.Json;
using UnityEditor.Experimental.GraphView;

[System.Serializable]
public class LatestBlock
{
    public string block_type;
    public int x;
    public int y;
    public int z;
    public int orientation;
}

[System.Serializable]
public class Env
{
    public LatestBlock latest_block;
    public bool block_conflict;
}

[System.Serializable]
public class Agent
{
    public int[] actions;
    public string explore_exploit;
    public float epsilon;
    public int eps_steps;
}

[System.Serializable]
public class Record
{
    public int episode;
    public int block;
    public Env env;
    public Agent agent;
    public float loss;
    public float reward;
    public bool terminal;
}

[System.Serializable]
public class Root
{
    public Record[] record;
}

public class VizResults : MonoBehaviour
{
    private string filePath = "../../results/2024-03-21_HCNN/episode_0_blocks_0-50_log.json";

    // Start is called before the first frame update
    void Start()
    {
        // Debug.Log("Hello world!");
        if (File.Exists(filePath))
        {
            // Deserialize the JSON data to a Root object
            Root rootData = JsonUtility.FromJson<Root>(File.ReadAllText(filePath));

            // Output the data to the console for verification
            foreach (Record record in rootData.record)
            {
                Debug.Log("Episode: " + record.episode);
                Debug.Log("Block: " + record.block);
                Debug.Log("Latest Block Type: " + record.env.latest_block.block_type);
                Debug.Log("Agent Actions: " + string.Join(", ", record.agent.actions));
                Debug.Log("Reward: " + record.reward);
                Debug.Log("Terminal: " + record.terminal);
            }
        }
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
