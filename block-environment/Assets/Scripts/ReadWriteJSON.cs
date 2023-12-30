using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.Linq;

using System.IO;
using System.Runtime.Serialization.Json;

using static Block;

public class Design
{
    public int ID;
    public string lastModified;
    public string lastAuthor;
    public string[] objectList;
}

public class ReadWriteJSON : MonoBehaviour
{
    public bool readMode = false;

    private string filePath = "design.json";

    private int designID = -1; // even designIDs are for the architect, odd designIDs are for the agent

    public bool waiting = false;

    //boolean for checking if JSON has been read
    private bool jsonRead = false;

    private bool idAssigned = false;

    // Start is called before the first frame update
    void Start()
    {
        WriteDesignJSON("user");
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
        DataContractJsonSerializer serializer = new DataContractJsonSerializer(typeof(Design));

        using (FileStream stream = new FileStream(filePath, FileMode.Create))
        {
            serializer.WriteObject(stream, newDesign);
        }

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
        
    }
}
