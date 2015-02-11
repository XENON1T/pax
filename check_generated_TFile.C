void check_generated_TFile() {

    TFile* f = TFile::Open("tree.root");

    cout << "Contents of file \"tree.root\":" << endl;
    f->ls();

    TTree* tree = (TTree*)f->Get("name_of_tree");

    cout << "Calling Print on tree:" << endl;
    tree->Print();

    cout << "Scan and print actual stored values: " << endl;
    tree->Scan("*");

    return;
}
