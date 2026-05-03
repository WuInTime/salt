use symbolica::LicenseManager;

fn main() {
    LicenseManager::request_hobbyist_license("Y Wu", "ywu86@ur.rochester.edu").unwrap();
}
