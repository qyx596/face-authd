fn main() {
    cc::Build::new()
        .cpp(true)
        .file("src/dlib_face.cpp")
        .flag("-std=c++14")
        .flag("-O2")
        .flag("-w") // suppress dlib's template warnings
        .compile("dlib_face_wrapper");

    println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
    println!("cargo:rustc-link-lib=dlib");
    // blas/lapack are only installed as versioned .so.3 on this system
    println!("cargo:rustc-link-arg=/usr/lib/x86_64-linux-gnu/libblas.so.3");
    println!("cargo:rustc-link-arg=/usr/lib/x86_64-linux-gnu/liblapack.so.3");
    println!("cargo:rustc-link-lib=sqlite3");
    println!("cargo:rustc-link-lib=pthread");

    println!("cargo:rerun-if-changed=src/dlib_face.cpp");
    println!("cargo:rerun-if-changed=src/dlib_face.h");
}
