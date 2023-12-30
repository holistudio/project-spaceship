using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using UnityEngine;

public class Block : MonoBehaviour
{
    public string type;
    public Vector3 position;
    public int orientation;
    public Vector3 color;
    public int sequenceIndex = -1;

    private bool modified = false;

    int GetOrientation()
    {
        int angle = (int) Math.Round(transform.rotation.eulerAngles.y);

        if (angle % 90 == 0)
        {
            if ((angle/90 % 2) == 0)
            {
                return 0;
            }
            else
            {
                return 1;
            }
        }
        return -1;
    }

    void UpdateProperties()
    {
        type = gameObject.name;
        position = transform.position;
        orientation = GetOrientation();
        Color rgba_color = gameObject.GetComponent<MeshRenderer>().material.color;
        color = new Vector3(rgba_color.r,rgba_color.g,rgba_color.b); 
    }
    
    // Start is called before the first frame update
    void Start()
    {
        UpdateProperties();
    }

    bool CheckModified()
    {
        Color rgba_color = gameObject.GetComponent<MeshRenderer>().material.color;
        Vector3 latest_color = new Vector3(rgba_color.r,rgba_color.g,rgba_color.b);
        if((transform.position == position) && (GetOrientation() == orientation) && (latest_color == color))
        {
            return false;
        }
        return true;
    }

    // Update is called once per frame
    void Update()
    {
        modified = CheckModified();
        if(modified)
        {
            UpdateProperties();
        }
    }
}
