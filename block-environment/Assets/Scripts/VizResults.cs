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
public class Step
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
    public Step[] record;
}

public class VizResults : MonoBehaviour
{
    public GameObject blockSet;
    private string filePath = "../../results/2024-03-21_HCNN/episode_0_blocks_0-50_log.json";

    Vector3 convertToUnityPosition(LatestBlock latestBlockData)
    {
        Vector3 unityPosition = new Vector3(0.0f, 0.0f, 0.0f);
        string blockType = latestBlockData.block_type;
        Vector3 gridPosition = new Vector3(latestBlockData.x,latestBlockData.y,latestBlockData.z);
        int orient = latestBlockData.orientation;

        if (blockType.Equals("2x1"))
        {
            if(orient == 0)
            {
                unityPosition.x = gridPosition.x + 1;
                unityPosition.y = gridPosition.y + 0.5f;
                unityPosition.z = gridPosition.z - 0.5f;
            }
            else
            {
                unityPosition.x = gridPosition.x + 0.5f;
                unityPosition.y = gridPosition.y + 0.5f;
                unityPosition.z = gridPosition.z + 1;
            }
        }
        else if (blockType.Equals("3x1"))
        {
            if(orient == 0)
            {
                unityPosition.x = gridPosition.x + 1.5f;
                unityPosition.y = gridPosition.y + 0.5f;
                unityPosition.z = gridPosition.z + 0.5f;
            }
            else
            {
                unityPosition.x = gridPosition.x + 0.5f;
                unityPosition.y = gridPosition.y + 0.5f;
                unityPosition.z = gridPosition.z + 1.5f;
            }
        }
        else if (blockType.Equals("4x1"))
        {
            if(orient == 0)
            {
                unityPosition.x = gridPosition.x + 2;
                unityPosition.y = gridPosition.y + 0.5f;
                unityPosition.z = gridPosition.z + 0.5f;
            }
            else
            {
                unityPosition.x = gridPosition.x + 0.5f;
                unityPosition.y = gridPosition.y + 0.5f;
                unityPosition.z = gridPosition.z + 2;
            }
        }
        else if (blockType.Equals("2x2"))
        {
            unityPosition.x = gridPosition.x + 1;
            unityPosition.y = gridPosition.y + 0.5f;
            unityPosition.z = gridPosition.z + 1;
        }
        else if (blockType.Equals("3x2"))
        {
            if(orient == 0)
            {
                unityPosition.x = gridPosition.x + 1.5f;
                unityPosition.y = gridPosition.y + 0.5f;
                unityPosition.z = gridPosition.z;
            }
            else
            {
                unityPosition.x = gridPosition.x + 1;
                unityPosition.y = gridPosition.y + 0.5f;
                unityPosition.z = gridPosition.z + 0.5f;
            }
        }
        else if (blockType.Equals("4x2"))
        {
            if(orient == 0)
            {
                unityPosition.x = gridPosition.x + 2;
                unityPosition.y = gridPosition.y + 0.5f;
                unityPosition.z = gridPosition.z;
            }
            else
            {
                unityPosition.x = gridPosition.x + 1;
                unityPosition.y = gridPosition.y + 0.5f;
                unityPosition.z = gridPosition.z + 1;
            }
        }
        return  unityPosition;
    }
    // Start is called before the first frame update
    void Start()
    {
        // Debug.Log("Hello world!");
        if (File.Exists(filePath))
        {
            // Deserialize the JSON data to a Root object
            Root rootData = JsonUtility.FromJson<Root>(File.ReadAllText(filePath));

            Step step = rootData.record[0];

            LatestBlock listBlockData = step.env.latest_block;

            Transform childTransform = blockSet.transform.Find(listBlockData.block_type);
            
            if (childTransform != null)
            {
                GameObject blockType = childTransform.gameObject;
                // Instantiate a copy of the original GameObject
                GameObject blockCopy = Instantiate(blockType);

                blockCopy.name = listBlockData.block_type;

                // Set the copy's parent to this game object
                blockCopy.transform.SetParent(transform);

                Quaternion blockRotation;

                if(listBlockData.orientation == 0)
                {
                    blockRotation = Quaternion.Euler(0, 0, 0);
                }
                else
                {
                    blockRotation = Quaternion.Euler(0, 90, 0);
                }

                Vector3 unityPosition = convertToUnityPosition(listBlockData);
                blockCopy.transform.SetLocalPositionAndRotation(unityPosition, blockRotation);
            }

            Debug.Log(step.loss);
            // Output the data to the console for verification
            // foreach (Step step in rootData.record)
            // {
            //     Debug.Log("Episode: " + step.episode);
            //     Debug.Log("Block: " + step.block);
            //     Debug.Log("Latest Block Type: " + step.env.latest_block.block_type);
            //     Debug.Log("Agent Actions: " + string.Join(", ", step.agent.actions));
            //     Debug.Log("Reward: " + step.reward);
            //     Debug.Log("Terminal: " + step.terminal);
            // }
        }
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
