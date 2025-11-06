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
    public bool block_conflict;
}

[System.Serializable]
public class Env
{
    public LatestBlock latest_agent_block;
    public LatestBlock latest_env_block;
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

public class VizEpisode : MonoBehaviour
{
    public GameObject blockSet;
    public int episode;
    public int blocksPerEpisode;
    private int startBlockIndex;
    private int endBlockIndex;
    public int stepIndex = 0;
    private string fileName;
    private string filePath;
    private string folderPath = "../../results/2025-10-09_env/";

    private Root rootData;
    private Step[] stepData;

    private bool upToDate = false;
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
                unityPosition.z = gridPosition.z + 0.5f;
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
                unityPosition.z = gridPosition.z + 1.0f;
            }
            else
            {
                unityPosition.x = gridPosition.x + 1.0f;
                unityPosition.y = gridPosition.y + 0.5f;
                unityPosition.z = gridPosition.z + 1.5f;
            }
        }
        else if (blockType.Equals("4x2"))
        {
            if(orient == 0)
            {
                unityPosition.x = gridPosition.x + 2.0f;
                unityPosition.y = gridPosition.y + 0.5f;
                unityPosition.z = gridPosition.z + 1.0f;
            }
            else
            {
                unityPosition.x = gridPosition.x + 1.0f;
                unityPosition.y = gridPosition.y + 0.5f;
                unityPosition.z = gridPosition.z + 2.0f;
            }
        }
        return  unityPosition;
    }
    // Start is called before the first frame update
    void Start()
    {
        startBlockIndex = 0;
        bool endOfEpisode = false;

        while (!endOfEpisode)
        {
            endBlockIndex = startBlockIndex + blocksPerEpisode - 1;

            string fileNameStart = $"episode_{episode}_blocks_{startBlockIndex}";
            fileName = $"{fileNameStart}-{endBlockIndex}_log.json";
            filePath = Path.Combine(folderPath, fileName);

            // check if any files in the folder start with the last json file for the episode
            // ex: find 'episode_0_blocks_1300-1311_log.json' instead of 'episode_0_blocks_1300-1349_log.json'
            
            if (!File.Exists(filePath))
            {
                // Get all files in the folder
                string[] files = Directory.GetFiles(folderPath);
                foreach (string tempFilePath in files)
                {
                    // Get the file name from the full path
                    string tempName = Path.GetFileName(tempFilePath);

                    // Check if the file name starts with the specified sequence
                    if (tempName.StartsWith(fileNameStart))
                    {
                        fileName = tempName;
                        filePath = Path.Combine(folderPath, fileName);
                    }
                }
            }

            if (File.Exists(filePath))
            {
                // Deserialize the JSON data to a Root object
                rootData = JsonUtility.FromJson<Root>(File.ReadAllText(filePath));
                stepData = rootData.record;

                foreach (Step step in rootData.record)
                {
                    if (step.env.latest_agent_block.block_conflict == false)
                    {
                        Debug.Log("Episode: " + step.episode);
                        Debug.Log("Block: " + step.block);
                        Debug.Log("Latest Agent Block Type: " + step.env.latest_agent_block.block_type);
                        Debug.Log("Latest Env Block Type: " + step.env.latest_env_block.block_type);
                        Debug.Log("Block Conflict: " + step.env.latest_agent_block.block_conflict);
                        string agentBlockType = step.env.latest_agent_block.block_type;
                        string envBlockType = step.env.latest_env_block.block_type;
                        if (!agentBlockType.Equals("None"))
                        {
                            addBlock(step, "agent");
                        }
                        if (!envBlockType.Equals("None"))
                        {
                            addBlock(step, "env");
                        }
                    }
                    
                }
                startBlockIndex += blocksPerEpisode;
            }
            else
            {
                endOfEpisode = true;
            }

        }
    }

    void addBlock(Step step, string blockAuthor)
    {
        if (blockAuthor.Equals("agent") || blockAuthor.Equals("env"))
        {
            LatestBlock listBlockData = step.env.latest_agent_block;
            
            if (blockAuthor.Equals("env"))
            {
                listBlockData = step.env.latest_env_block;
            }

            Transform childTransform = blockSet.transform.Find(listBlockData.block_type);
            
            if (childTransform != null)
            {
                GameObject blockType = childTransform.gameObject;
                // Instantiate a copy of the original GameObject
                GameObject blockCopy = Instantiate(blockType);

                blockCopy.name = listBlockData.block_type + "_" + step.block.ToString();

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

        }
        Debug.Log("Error: provide either 'agent' or 'env' as blockAuthor.");
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
