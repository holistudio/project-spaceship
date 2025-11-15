using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using System.Linq;

using System.IO;
using System.Text.Json;
using UnityEditor.Experimental.GraphView;
using Unity.VisualScripting;


public class VizResults : MonoBehaviour
{
    public GameObject blockSet;
    public int stepIndex = 0;
    private string filePath = "../../results/2025-11-13_env/episode_0_blocks_100-149_log.json";

    private Root rootData;

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
        // Debug.Log("Hello world!");
        if (File.Exists(filePath))
        {
            // Deserialize the JSON data to a Root object
            rootData = JsonUtility.FromJson<Root>(File.ReadAllText(filePath));
            // Debug.Log(step.loss);
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
            for (int i = 0; i < stepIndex+1; i++)
            {
                addBlock(i);
                Debug.Log("Step: " + i);
            }
        }
    }

    void addBlock(int i)
    {
        Step step = rootData.record[i];

        // LatestBlock listBlockData = step.env.latest_agent_block;
        LatestBlock listBlockData = step.env.latest_block;

        Transform childTransform = blockSet.transform.Find(listBlockData.block_type);
        
        if (childTransform != null)
        {
            GameObject blockType = childTransform.gameObject;
            // Instantiate a copy of the original GameObject
            GameObject blockCopy = Instantiate(blockType);

            blockCopy.name = listBlockData.block_type;
            blockCopy.GetComponent<Block>().sequenceIndex = step.block;

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

    // Update is called once per frame
    void Update()
    {
        // Check if the right arrow key is pressed
        if (Input.GetKeyDown(KeyCode.RightArrow))
        {
            stepIndex++;
            upToDate = false;
            Debug.Log("Step: " + stepIndex);
        }

        if (!upToDate)
        {
            addBlock(stepIndex);
            upToDate = true;
        }
        
    }
}
