using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.Linq;

using System.IO;
using System.Text.Json;

using static Block;

public class Design
{
    public int ID;
    public string lastModified;
    public string lastAuthor;
    public string[] objectList;
}

public class BlockData
{
    public string type;
    public Vector3 position;
    public int orientation;
    public Vector3 color;
    public int sequenceIndex;
}

public class ReadWriteJSON : MonoBehaviour
{
    public GameObject blockSet;

    private string filePath = "design_shapenet_x2.json";

    private int designID = -1; // even designIDs are for the architect, odd designIDs are for the agent

    public bool waiting = true;

    //boolean for checking if JSON has been read
    private bool jsonRead = false;

    private bool idAssigned = false;

    // Start is called before the first frame update
    void Start()
    {
        ReadDesignJSON();
    }

    Design recordDesignBlocks()
    {
        Design newDesign = new Design();

        newDesign.ID = designID;

        int childCount = transform.childCount;
        int blockCount = 0;

        for (int i = 0; i < childCount; i++)
        {
            GameObject child = transform.GetChild(i).gameObject;

            Block thisBlock = child.GetComponent<Block>();

            if (thisBlock != null)
            {
                blockCount++;
            }
        }

        newDesign.objectList = new string[blockCount];

        int b_i = 0;
        for (int i = 0; i < childCount; i++)
        {
            GameObject child = transform.GetChild(i).gameObject;

            Block thisBlock = child.GetComponent<Block>();
            if (thisBlock != null)
            {
                newDesign.objectList[b_i] = JsonUtility.ToJson(thisBlock);
                b_i++;
            }
        }

        return newDesign;
    }

    public void ReadDesignJSON()
    {
        if (File.Exists(filePath))
        {
            // Load design from the JSON file
            Design currentDesign = JsonUtility.FromJson<Design>(File.ReadAllText(filePath));

            // If the Design ID has changed, update objects based one the new design
            if(currentDesign.ID > designID)
            {
                // waiting = false;
                designID = currentDesign.ID;

                //load new design
                for (int i = 0; i < currentDesign.objectList.Length; i++)
                {
                    BlockData listBlockData = JsonUtility.FromJson<BlockData>(currentDesign.objectList[i]);

                    Transform childTransform;
                    if(listBlockData.type.Equals("1x1"))
                    {
                        // copy block type in block 
                        childTransform = blockSet.transform.Find("Cube");
                    }
                    else
                    {
                        // copy block type in block 
                        childTransform = blockSet.transform.Find(listBlockData.type);
                    }
                    

                    if (childTransform != null)
                    {
                        GameObject blockType = childTransform.gameObject;
                        // Instantiate a copy of the original GameObject
                        GameObject blockCopy = Instantiate(blockType);

                        blockCopy.name = listBlockData.type;

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
                        blockCopy.transform.SetLocalPositionAndRotation(listBlockData.position,blockRotation);
                    }
                }
            }
        }
    }
    public void WriteDesignJSON(string author)
    {
        if(author.Equals("user"))
        {
            designID++; 
        }

        Design newDesign = recordDesignBlocks();

        newDesign.lastAuthor = author;

        //set lastModfied to current time
        DateTime currentTime = DateTime.Now;

        // Format the current time as a string
        string formattedTime = currentTime.ToString("yyyy-MM-dd HH:mm:ss");

        newDesign.lastModified = formattedTime;

        // Serialize the object to JSON
        string json = JsonUtility.ToJson(newDesign);

        File.WriteAllText(filePath, json);

        if(author.Equals("user"))
        {
            waiting = true;
            jsonRead = false;
            print("Design updated to iteration "+designID.ToString()+". Waiting for Agent...");
        }
    }
    // Update is called once per frame
    void Update()
    {
        if(!waiting)
        {
            // WriteDesignJSON("user");
        }
    }
}
