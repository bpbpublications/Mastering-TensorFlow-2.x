

package main

import (
	"encoding/json"
	"log"
	"net/http"
)

func main() {

	http.HandleFunc("/", hello)

	log.Print("Go server listening on port 8080")
	log.Fatal(http.ListenAndServe(":8080", nil))

}

func hello(w http.ResponseWriter, r *http.Request) {

	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}

	dictionary := make(map[string]string)
	dictionary["Welcome to Developing Cloud Native Apps on GCP"] = "Success you have just completed the tutorial!!"

	json, _ := json.Marshal(dictionary)

	w.Header().Set("Content-Type", "application/json")
	w.Write(json)

}
